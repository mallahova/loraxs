import logging
import math
import types
from typing import Any, Dict, List, Optional, Tuple

import peft
import torch
from peft import PeftConfig, PeftModel
from peft.import_utils import is_bnb_available
from peft.utils import _get_submodules
from torch import Tensor
from torch.nn import Linear, init
from tqdm import tqdm

from utils.adaptive.mask import WeightMaskingLinear
from utils.latent_utils import forward_latent, get_delta_weight
from utils.svd_utils import get_linear_rec_svd

logger = logging.getLogger(__name__)


def get_replacement_module(
    weight: torch.Tensor, module_name: str, type_: str, writer: Optional[Any], reconstruct_config: Dict[str, Any]
) -> Tuple[torch.Tensor, torch.Tensor]:
    cfg = reconstruct_config[type_]
    if type_ == "svd":
        reconstructed_matrix, enc, dec = get_linear_rec_svd(
            weight.cpu().detach().numpy(),
            cfg["rank"],
            cfg["n_iter"],
            cfg["random_state"],
        )
        final_enc = torch.tensor(enc, dtype=weight.dtype, device=weight.device)
        final_dec = torch.tensor(dec, dtype=weight.dtype, device=weight.device)
    else:
        raise NotImplementedError(f"{type_} is currently not supported.")
    return final_enc, final_dec


def init_module_weights(target_module: torch.nn.Linear, sigma: float) -> None:
    # Initialize weights with Gaussian distribution
    torch.nn.init.normal_(target_module.weight, mean=0, std=sigma)
    if hasattr(target_module, "bias") and target_module.bias is not None:
        # Set bias to zeros
        torch.nn.init.zeros_(target_module.bias)


def replace_module_weights(target_module: torch.nn.Module, new_weight: torch.Tensor) -> None:
    device = target_module.weight.device
    target_module.weight = torch.nn.Parameter(new_weight)

    # dispatch to correct device
    for name, module in target_module.named_modules():
        if "lora_" in name:
            module.to(device)


def update_decoder_weights(target_module: torch.nn.Module, new_weight: torch.Tensor) -> None:
    device = target_module.weight.device
    with torch.no_grad():
        target_module.weight.copy_(new_weight)

    # dispatch to correct device
    for name, module in target_module.named_modules():
        if "lora_" in name:
            module.to(device)


def kaiming_uniform_init_lower_half(matrix: torch.Tensor) -> torch.Tensor:
    rows, _ = matrix.size()
    init.kaiming_uniform_(matrix[math.ceil(rows / 2) :, :], a=math.sqrt(5))
    return matrix


def kaiming_uniform_init(matrix: torch.Tensor) -> torch.Tensor:
    init.kaiming_uniform_(matrix, a=math.sqrt(5))
    return matrix


def find_and_initialize(
    model: PeftModel,
    peft_config: Dict[str, PeftConfig],
    adapter_name: str,
    reconstr_type: str,
    reconstruct_config: Dict[str, Any],
    writer: Optional[Any],
    rank_allocation_args: Any,
) -> None:
    (
        adaptive_rank_allocation,
        debug_mode,
        half_init_dec,
        is_target_modules_in_base_model,
        key_list,
        lora_config,
        r_squared,
        rank_allocation_weights_init,
        reconstruction_mode,
        replacement_module_random_init,
        target_modules_count,
    ) = extract_config(adapter_name, model, peft_config, reconstruct_config)
    print("Iterating through model's specified modules to initialize A/B matrices.")
    for key in tqdm(key_list):
        if debug_mode and target_modules_count > 2:
            break
        target_module_found = any(key.endswith(target_key) for target_key in lora_config.target_modules)
        if target_module_found:
            target_modules_count += 1
            if not is_target_modules_in_base_model:
                is_target_modules_in_base_model = True
            _, target, target_name = _get_submodules(model, key)

            if reconstruction_mode == "separated":
                replacement_encoder_weight, replacement_decoder_weight = get_replacement_module(
                    weight=target.weight.T,
                    module_name=key,
                    type_=reconstr_type,
                    writer=writer,
                    reconstruct_config=reconstruct_config,
                )

                if not isinstance(target, peft.tuners.lora.Linear):
                    raise NotImplementedError("Only initialization for peft.tuners.lora.Linear type is implemented.")
                else:
                    initialize_lora_weights(
                        adaptive_rank_allocation,
                        half_init_dec,
                        lora_config,
                        r_squared,
                        rank_allocation_args,
                        replacement_decoder_weight,
                        replacement_encoder_weight,
                        replacement_module_random_init,
                        target,
                    )
            else:
                raise NotImplementedError("The only supported mode is: separated.")

    if not is_target_modules_in_base_model:
        raise ValueError(
            f"Target modules {lora_config.target_modules} not found in the base model. "
            f"Please check the target modules and try again."
        )

    if adaptive_rank_allocation:
        initialize_adaptive(model, rank_allocation_weights_init, target_modules_count)


def initialize_lora_weights(
    adaptive_rank_allocation: bool,
    half_init_dec: bool,
    lora_config: PeftConfig,
    r_squared: bool,
    rank_allocation_args: Any,
    replacement_decoder_weight: Tensor,
    replacement_encoder_weight: Tensor,
    replacement_module_random_init: bool,
    target: Linear,
) -> None:
    if half_init_dec:
        kaiming_uniform_init_lower_half(replacement_decoder_weight)
    if replacement_module_random_init:
        kaiming_uniform_init(replacement_encoder_weight)
        kaiming_uniform_init(replacement_decoder_weight)
    replace_module_weights(target.lora_B.default, replacement_decoder_weight.T)
    if r_squared:
        target.forward = types.MethodType(forward_latent, target)
        target.get_delta_weight = types.MethodType(get_delta_weight, target)
        replace_module_weights(target.lora_A.default, replacement_encoder_weight.T)

        if adaptive_rank_allocation:
            target.default_lora_latent_mapping = WeightMaskingLinear(
                lora_config.r,
                lora_config.r,
                rank_min=rank_allocation_args.rank_min,
                rank_max=rank_allocation_args.rank_max,
                tau=rank_allocation_args.tau,
                bias=False,
            )
        else:
            target.default_lora_latent_mapping = torch.nn.Linear(lora_config.r, lora_config.r, bias=False)

        init_module_weights(target.default_lora_latent_mapping, sigma=0.00001)
        target.default_lora_latent_mapping.to(target.lora_A.default.weight.device)
        target.lora_A.default.weight.requires_grad = False  # only the r*r matrix will be tuned
        target.lora_B.default.weight.requires_grad = False  # only the r*r matrix will be tuned

    else:
        init_module_weights(target.lora_A.default, sigma=0.00001)


def extract_config(
    adapter_name: str, model: PeftModel, peft_config: Dict[str, PeftConfig], reconstruct_config: Dict[str, Any]
) -> Tuple[bool, bool, bool, bool, List[str], PeftConfig, bool, Optional[str], str, bool, int]:
    debug_mode = reconstruct_config.get("debug_mode", False)
    adaptive_rank_allocation = reconstruct_config.get("adaptive_rank_allocation", True)
    rank_allocation_weights_init = reconstruct_config.get("rank_allocation_weights_init")

    half_init_dec = reconstruct_config["half_init_dec"]
    replacement_module_random_init = reconstruct_config["replacement_module_random_init"]
    reconstruction_mode = reconstruct_config["reconstr_mode"]
    lora_config = peft_config[adapter_name]
    r_squared = reconstruct_config["r_squared"]  # whether using r*r matrix between lora_A and lora_B or not
    loaded_in_8bit = getattr(model, "is_loaded_in_8bit", False)
    if loaded_in_8bit and not is_bnb_available():
        raise ImportError(
            "To use Lora with 8-bit quantization, please install the `bitsandbytes` package. "
            "You can install it with `pip install bitsandbytes`."
        )
    is_target_modules_in_base_model = False
    key_list = [key for key, _ in model.named_modules()]
    target_modules_count = 0
    assert not isinstance(lora_config.target_modules, str)
    return (
        adaptive_rank_allocation,
        debug_mode,
        half_init_dec,
        is_target_modules_in_base_model,
        key_list,
        lora_config,
        r_squared,
        rank_allocation_weights_init,
        reconstruction_mode,
        replacement_module_random_init,
        target_modules_count,
    )


def initialize_adaptive(
    model: PeftModel, rank_allocation_weights_init: Optional[str], target_modules_count: int
) -> None:
    if rank_allocation_weights_init == "uniform":
        param = torch.nn.Parameter(torch.zeros(target_modules_count))
    elif rank_allocation_weights_init == "quadratic":
        center = target_modules_count // 2  # Center of the distribution (mean)
        sigma = target_modules_count / (2 * (10) ** 0.5)
        # Generate evenly spaced points
        x = torch.linspace(0, target_modules_count - 1, steps=target_modules_count)
        # Compute Gaussian function
        param = torch.nn.Parameter(torch.exp(-((x - center) ** 2) / (2 * sigma**2)))
    elif rank_allocation_weights_init == "left_skewed":
        center = 3 * target_modules_count // 4  # Center the peak at 3/4n
        sigma_left = 3 * target_modules_count / (4 * (10) ** 0.5)  # Standard deviation for the left side
        sigma_right = 1 * target_modules_count / (4 * (10) ** 0.5)  # Standard deviation for the right side
        # Generate evenly spaced points
        x = torch.linspace(0, target_modules_count - 1, steps=target_modules_count)
        # Compute left-skewed distribution
        param = torch.nn.Parameter(
            torch.where(
                x <= center,
                torch.exp(-((x - center) ** 2) / (2 * sigma_left**2)),
                torch.exp(-((x - center) ** 2) / (2 * sigma_right**2)),
            )
        )
    else:
        param = torch.nn.Parameter(torch.randn(target_modules_count))
    model.register_parameter(name="rank_allocation_weights", param=param)
