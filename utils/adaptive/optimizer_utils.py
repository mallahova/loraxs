import logging
import math
from typing import Tuple, Union

import torch
from datasets import Dataset
from peft import PeftMixedModel, PeftModel
from torch.optim import AdamW
from torch.optim.lr_scheduler import LRScheduler
from transformers import TrainingArguments, get_constant_schedule, get_linear_schedule_with_warmup

from utils.adaptive.args import ModelArguments, RankAllocaionArguments

logger = logging.getLogger(__name__)


def setup_optimizer_and_scheduler(
    model: Union[PeftModel, PeftMixedModel],
    model_args: ModelArguments,
    rank_allocation_args: RankAllocaionArguments,
    train_dataset: Dataset,
    training_args: TrainingArguments,
) -> Tuple[int, int, AdamW, LRScheduler]:
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [i[1] for i in model.named_parameters() if "classifier" in i[0]],
                "lr": model_args.cls_learning_rate,
            },
            {
                "params": [
                    i[1]
                    for i in model.named_parameters()
                    if "classifier" not in i[0] and "rank_allocation_weights" not in i[0]
                ],
                "lr": training_args.learning_rate,
            },
            {
                "params": [model.rank_allocation_weights],
                "lr": rank_allocation_args.rank_allocation_lr,
            },
        ]
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataset) / (training_args.gradient_accumulation_steps * training_args.per_device_train_batch_size)
    )
    max_train_steps = training_args.num_train_epochs * num_update_steps_per_epoch
    if model_args.lr_scheduler == "linear_schedule_with_warmup":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.06 * max_train_steps),
            num_training_steps=max_train_steps,
        )
    elif model_args.lr_scheduler == "constant_schedule":
        scheduler = get_constant_schedule(optimizer)
    else:
        raise ValueError(f"Unknown scheduler {model_args.lr_scheduler}.")
    return max_train_steps, num_update_steps_per_epoch, optimizer, scheduler
