import logging

import torch
from transformers import (
    Trainer,
)

from utils.adaptive.initialization_utils import WeightMaskingLinear
from utils.adaptive.schedulers import get_alpha_scheduler

logger = logging.getLogger(__name__)


class RankMaskingTrainer(Trainer):
    """
    Custom Trainer class to apply random or fixed masking during training and evaluation.
    """

    def __init__(
        self,
        rank_allocation_args,
        max_train_steps: int,
        num_update_steps_per_epoch: int,
        rank_min=15,
        rank_max=25,
        memory_start=20,
        memory_end=25,
        alpha_min=0.5,
        alpha_max=3,
        epochs_memory_start=None,
        epochs_memory_start_to_end=None,
        epochs_rank_discrete=0,
        *args,
        **kwargs,
    ):
        """
        :param rank_min: The minimum rank that can be assigned for a parameter matrix.
        :param rank_max: The maximum rank that can be assigned for a parameter matrix.
        Each of the trainable parameters is initialized with rank_max*rank_max matrices.
        :param memory_start: The total number of parameters that can be allocated at the start of the training.
        :param memory_end: The total number of parameters that can be allocated at the end of the training.
        Alpha - a parameter that impacts discretization of the masks applied to the trainable parameters.
        :param alpha_min: Minimum alpha value. Initial alpha.
        :param alpha_max: Maximum alpha value. Final alpha.
        :param epochs_memory_start: Number of epochs to keep the initial memory static.
        :param epochs_memory_start_to_end: Number of epochs to linearly increase/decrease the memory from memory_start to memory_end.
        The number of epochs to keep memory_end static is calculated as total_epochs - epochs_memory_start - epochs_memory_start_to_end.
        :param epochs_rank_discrete: Number of final epochs to keep the rank allocation discrete.
        """
        super().__init__(*args, **kwargs)
        self.rank_min = rank_min
        self.rank_max = rank_max

        self.memory = memory_start
        if memory_start != memory_end:
            self.memory_update = (memory_end - memory_start) / (epochs_memory_start_to_end * num_update_steps_per_epoch)
            self.epochs_memory_start = epochs_memory_start
            self.epochs_memory_start_to_end = epochs_memory_start_to_end
        else:
            self.memory_update = None
            self.epochs_memory_start = 0
            self.epochs_memory_start_to_end = 0

        self.alpha = alpha_min
        self.alpha_max = alpha_max
        self.alpha_scheduler = get_alpha_scheduler(
            rank_allocation_args.alpha_scheduler, alpha_min, alpha_max, max_train_steps
        )
        self.epochs_rank_discrete = epochs_rank_discrete

        self.rank_allocation = None

    def update_memory(self):
        self.memory += self.memory_update

    def training_step(self, model, inputs):
        # check if the current epoch should use discrete rank allocation
        current_epoch = int(self.state.epoch or 0)
        total_epochs = int(self.args.num_train_epochs)
        is_rank_discrete_epoch = current_epoch + self.epochs_rank_discrete == total_epochs

        self.rank_allocation = self._get_rank_allocation(model.rank_allocation_weights, self.rank_min, self.memory)
        if not is_rank_discrete_epoch:
            self._set_rank_mask(model, self.rank_allocation, self.alpha)
        else:
            self.rank_allocation = torch.round(self.rank_allocation).int()  # discretize the rank allocation
            self._set_rank_mask(model, self.rank_allocation, None)
        loss = super().training_step(model, inputs)
        self.log_rank_allocation()

        # check if the memory should be updated
        if (
            self.memory_update is not None
            and current_epoch >= self.epochs_memory_start
            and current_epoch < self.epochs_memory_start + self.epochs_memory_start_to_end
        ):
            self.update_memory()

        self.alpha = self.alpha_scheduler.step(self.alpha)
        return loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        Evaluate the model with masks of varying sizes.
        """
        self.rank_allocation = self._get_rank_allocation(self.model.rank_allocation_weights, self.rank_min, self.memory)
        # set discrete rank allocation
        self._set_rank_mask(self.model, self.rank_allocation, None)
        return super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

    def log_rank_allocation(self):
        """Log custom metrics to WandB."""
        self.log(
            {
                "alpha": self.alpha,
                "memory": self.memory,
                **{f"rank_allocation_{i}": rank for i, rank in enumerate(self.rank_allocation.tolist())},
            }
        )

    @classmethod
    def _set_rank_mask(cls, model, rank_allocation: torch.Tensor, alpha):
        """
        Set the mask for all WeightMaskingLinear layers in the model.
        Setting alpha to None discretizes the rank allocation.
        """
        ind = 0
        for _, module in model.named_modules():
            if isinstance(module, WeightMaskingLinear):
                s = rank_allocation[ind]
                ind += 1
                module.set_mask(s, alpha, ind)

    @classmethod
    def _get_rank_allocation(cls, w, rank_min, memory_size):
        """
        Get the rank allocation for each trainable parameter matrix.
        :param w: The rank allocation weights.
        :param rank_min: The minimum rank that can be assigned for a parameter
            matrix.
        :param memory_size: The total number of parameters that can be allocated.
        :return: Tensor of shape (N,) where N is the number of parameter
            matrices and each element is the rank allocated to that parameter
            matrix.
        """
        n = len(w)
        return torch.sqrt(rank_min**2 + (memory_size - n * rank_min**2) * torch.softmax(w, dim=0))
