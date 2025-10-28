import math
import types

import peft
import torch
from peft.import_utils import is_bnb_available
from peft.utils import _get_submodules
from torch.nn import init
from tqdm import tqdm

from .latent_utils import get_delta_weight, forward_latent
from .svd_utils import get_linear_rec_svd

import wandb
import matplotlib.pyplot as plt


class WeightMaskingLinear(torch.nn.Linear):
    """
    A Linear layer with dynamic masking applied to its weights.
    """

    def __init__(self, in_features, out_features, rank_min, rank_max, tau=0, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer("ranks", torch.arange(rank_min, rank_max + 1))
        self.rank_min = rank_min
        self.rank_max = rank_max
        self.tau = tau
        self.s = None
        self.alpha = None
        self.index = None

    def log_matrix_to_wandb(self, matrix):
        fig = plt.figure(figsize=(6, 6))
        plt.imshow(matrix, cmap="gray_r", interpolation="none", vmin=0, vmax=matrix.max())
        plt.colorbar(label="Value")

        num_rows, num_cols = matrix.shape
        plt.xticks(ticks=range(num_cols), labels=range(num_cols))
        plt.yticks(ticks=range(num_rows), labels=range(num_rows))

        plt.xlabel("Columns")
        plt.ylabel("Rows")
        plt.title(f"Rank: {self.s:.2f}, alpha: {self.alpha:.2f}")

        if wandb.run is not None:
            wandb.log({f"mask_{self.index}": wandb.Image(fig)})
        plt.close(fig)


    def _create_mask(self, p):
        mask=torch.zeros_like(self.weight)
        for i in range(self.rank_min, self.rank_max+1):
            tensor = torch.zeros_like(self.weight, device=self.weight.device, requires_grad=False)
            tensor[:i, :i] = 1
            mask+= p[i - self.rank_min] * tensor
        return mask

    def _create_discrete_mask(self):
        mask=torch.zeros_like(self.weight)
        discrete_rank = int(torch.round(self.s).item())
        mask[:discrete_rank, :discrete_rank] = 1
        return mask

    def _prob_dist(self, s,alpha):
        logits=-alpha*torch.log(1+(s-self.ranks)**2)
        gumbel_dist = torch.distributions.Gumbel(0,1)
        gumbel_vector = gumbel_dist.sample((self.rank_max+1-self.rank_min,)).to(self.weight.device)
        return torch.softmax(logits+self.tau*gumbel_vector,dim=0)

    def set_mask(self, s, alpha, index: int = 0):
        self.s=s
        self.alpha=alpha
        self.index=index


    def forward(self, input):
        """
        Apply the mask and compute the forward pass.
        """
        if self.s is not None:
            if self.alpha is not None:
                p=self._prob_dist(self.s,self.alpha).to(self.weight.device)
                mask=self._create_mask(p)
                self.log_matrix_to_wandb(mask.cpu().detach().numpy())
            else:
                mask=self._create_discrete_mask()
            masked_weight = self.weight * mask
        else:
            masked_weight = self.weight
        return torch.nn.functional.linear(input, masked_weight, self.bias)


def set_rank_mask(model, rank_allocation: torch.Tensor, alpha):
    """
    Set the mask for all WeightMaskingLinear layers in the model.
    Setting alpha to None discretizes the rank allocation.
    """
    ind=0
    for _, module in model.named_modules():
        if isinstance(module, WeightMaskingLinear):
            s=rank_allocation[ind]
            ind+=1
            module.set_mask(s, alpha, ind)



def get_replacement_module(weight, module_name, type, writer, reconstruct_config):
    cfg = reconstruct_config[type]
    if type == "svd":
        reconstructed_matrix, enc, dec = get_linear_rec_svd(
            weight.cpu().detach().numpy(),
            cfg["rank"],
            cfg["n_iter"],
            cfg["random_state"],
        )
        final_enc = torch.tensor(enc, dtype=weight.dtype, device=weight.device)
        final_dec = torch.tensor(dec, dtype=weight.dtype, device=weight.device)
    else:
        raise NotImplementedError(f"{type} is currently not supported.")
    return final_enc, final_dec


def init_module_weights(target_module: torch.nn.Linear, sigma: float):
    # Initialize weights with Gaussian distribution
    torch.nn.init.normal_(target_module.weight, mean=0, std=sigma)
    if hasattr(target_module, "bias"):
        # Set bias to zeros
        if target_module.bias is not None:
            torch.nn.init.zeros_(target_module.bias)


def replace_module_weights(target_module, new_weight):
    device = target_module.weight.device
    target_module.weight = torch.nn.Parameter(new_weight)

    # dispatch to correct device
    for name, module in target_module.named_modules():
        if "lora_" in name:
            module.to(device)


def update_decoder_weights(target_module, new_weight):
    device = target_module.weight.device
    with torch.no_grad():
        target_module.weight.copy_(new_weight)

    # dispatch to correct device
    for name, module in target_module.named_modules():
        if "lora_" in name:
            module.to(device)


def kaiming_uniform_init_lower_half(matrix: torch.tensor):
    rows, _ = matrix.size()
    init.kaiming_uniform_(matrix[math.ceil(rows / 2) :, :], a=math.sqrt(5))
    return matrix


def kaiming_uniform_init(matrix: torch.tensor):
    init.kaiming_uniform_(matrix, a=math.sqrt(5))
    return matrix


def find_and_initialize(
    model, peft_config, adapter_name, reconstr_type, reconstruct_config, writer, rank_allocation_args
):
    """
    :param adapter_name: options: 'default'
    :param reconstr_type: options: 'svd'
    """
    debug_mode = reconstruct_config.get("debug_mode", False)
    adaptive_rank_allocation = reconstruct_config.get("adaptive_rank_allocation", True)
    rank_allocation_weights_init = reconstruct_config.get("rank_allocation_weights_init", None)

    half_init_dec = reconstruct_config["half_init_dec"]
    replacement_module_random_init = reconstruct_config[
        "replacement_module_random_init"
    ]
    reconstruction_mode = reconstruct_config["reconstr_mode"]
    lora_config = peft_config[adapter_name]
    r_squared = reconstruct_config[
        "r_squared"
    ]  # whether using r*r matrix between lora_A and lora_B or not
    loaded_in_8bit = getattr(model, "is_loaded_in_8bit", False)
    if loaded_in_8bit and not is_bnb_available():
        raise ImportError(
            "To use Lora with 8-bit quantization, please install the `bitsandbytes` package. "
            "You can install it with `pip install bitsandbytes`."
        )
    is_target_modules_in_base_model = False
    key_list = [key for key, _ in model.named_modules()]
    target_modules_count=0
    assert not isinstance(lora_config.target_modules, str)
    print("Iterating through model's specified modules to initialize A/B matrices.")
    for key in tqdm(key_list):
        if debug_mode and target_modules_count>2: break
        target_module_found = any(
            key.endswith(target_key) for target_key in lora_config.target_modules
        )
        if target_module_found:
            target_modules_count+=1
            if not is_target_modules_in_base_model:
                is_target_modules_in_base_model = True
            _, target, target_name = _get_submodules(model, key)

            if reconstruction_mode == "separated":
                replacement_encoder_weight, replacement_decoder_weight = (
                    get_replacement_module(
                        weight=target.weight.T,
                        module_name=key,
                        type=reconstr_type,
                        writer=writer,
                        reconstruct_config=reconstruct_config,
                    )
                )

                if not isinstance(target, peft.tuners.lora.Linear):
                    raise NotImplementedError(
                        "Only initialization for peft.tuners.lora.Linear type is implemented."
                    )
                    # TODO implement for Linear8bitLt
                else:
                    if half_init_dec:
                        kaiming_uniform_init_lower_half(replacement_decoder_weight)
                    if replacement_module_random_init:
                        kaiming_uniform_init(replacement_encoder_weight)
                        kaiming_uniform_init(replacement_decoder_weight)
                    replace_module_weights(
                        target.lora_B.default, replacement_decoder_weight.T
                    )
                    if r_squared:
                        target.forward = types.MethodType(forward_latent, target)
                        target.get_delta_weight = types.MethodType(
                            get_delta_weight, target
                        )
                        replace_module_weights(
                            target.lora_A.default, replacement_encoder_weight.T
                        )

                        if adaptive_rank_allocation:
                            target.default_lora_latent_mapping = WeightMaskingLinear(
                                lora_config.r, lora_config.r, rank_min=rank_allocation_args.rank_min, rank_max=rank_allocation_args.rank_max, tau=rank_allocation_args.tau, bias=False
                            )
                        else:
                            target.default_lora_latent_mapping = torch.nn.Linear(
                                lora_config.r, lora_config.r, bias=False
                            )

                        init_module_weights(
                            target.default_lora_latent_mapping, sigma=0.00001
                        )
                        target.default_lora_latent_mapping.to(
                            target.lora_A.default.weight.device
                        )
                        target.lora_A.default.weight.requires_grad = (
                            False  # only the r*r matrix will be tuned
                        )
                        target.lora_B.default.weight.requires_grad = (
                            False  # only the r*r matrix will be tuned
                        )

                    else:
                        init_module_weights(target.lora_A.default, sigma=0.00001)

            else:
                raise NotImplementedError("The only supported mode is: separated.")

    if not is_target_modules_in_base_model:
        raise ValueError(
            f"Target modules {lora_config.target_modules} not found in the base model. "
            f"Please check the target modules and try again."
        )

    if adaptive_rank_allocation:
        if rank_allocation_weights_init == "uniform":
            param=torch.nn.Parameter(torch.zeros(target_modules_count))
        elif rank_allocation_weights_init == "quadratic":
            center = target_modules_count // 2  # Center of the distribution (mean)
            sigma = target_modules_count / (2 * (10) ** 0.5)
            # Generate evenly spaced points
            x = torch.linspace(0, target_modules_count - 1, steps=target_modules_count)
            # Compute Gaussian function
            param=torch.nn.Parameter(torch.exp(-((x - center) ** 2) / (2 * sigma ** 2)))
        elif rank_allocation_weights_init=="left_skewed":
            center = 3 * target_modules_count // 4  # Center the peak at 3/4n
            sigma_left = 3*target_modules_count / (4 * (10) ** 0.5)  # Standard deviation for the left side
            sigma_right = 1*target_modules_count / (4 * (10) ** 0.5)  # Standard deviation for the right side
            # Generate evenly spaced points
            x = torch.linspace(0, target_modules_count - 1, steps=target_modules_count)
            # Compute left-skewed distribution
            param=torch.nn.Parameter(torch.where(
                x <= center,
                torch.exp(-((x - center) ** 2) / (2 * sigma_left ** 2)),
                torch.exp(-((x - center) ** 2) / (2 * sigma_right ** 2))
            ))
        else:
            param=torch.nn.Parameter(torch.randn(target_modules_count))
        model.register_parameter(name='rank_allocation_weights', param=param)
