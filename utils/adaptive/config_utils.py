import datetime
import logging
import os
import sys
from typing import Any, Dict, Optional, Tuple

import yaml
from peft import LoraConfig, PromptLearningConfig
from torch.utils.tensorboard import SummaryWriter
from transformers import HfArgumentParser, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from utils.adaptive.args import DataTrainingArguments, ModelArguments, RankAllocaionArguments

logger = logging.getLogger(__name__)


def check_last_checkpoint(training_args) -> Optional[str]:
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    return last_checkpoint


def parse_arguments() -> Tuple[Any, Any, Any, Any]:
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, RankAllocaionArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, rank_allocation_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args, rank_allocation_args = parser.parse_args_into_dataclasses()
    return data_args, model_args, rank_allocation_args, training_args


def setup_output_dirs(
    data_args, model_args, peft_config: LoraConfig, training_args
) -> Tuple[SummaryWriter, Dict[Any, Any], Any, str, Any]:
    now = datetime.datetime.now()
    now = now.strftime("%Y-%m-%dT%H:%M:%S") + ("-%02d" % (now.microsecond / 10000))

    adapter_name = "default"
    peft_config_dict = {}
    if not isinstance(peft_config, PromptLearningConfig):
        peft_config_dict[adapter_name] = peft_config

    reconstr_config, reconstr_type = load_reconstruction_config(adapter_name, peft_config_dict)

    training_args.output_dir = (
        f"{training_args.output_dir}/{model_args.model_name_or_path}/{data_args.task_name}/"
        f"LoRA_init_{reconstr_type}_rank_{peft_config_dict[adapter_name].r}_lr_{training_args.learning_rate}_"
        f"clslr_{model_args.cls_learning_rate}_seed_{training_args.seed}/output_{now}"
    )
    os.makedirs(training_args.output_dir)

    log_dir = f"{training_args.output_dir}/tb_logs"
    os.makedirs(log_dir, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=log_dir)
    return peft_config_dict, adapter_name, reconstr_type, reconstr_config, tb_writer


def load_reconstruction_config(adapter_name: str, peft_config_dict: Dict[Any, Any]) -> Tuple[Any, Any]:
    with open("config/reconstruct_config.yaml", "r") as stream:
        reconstr_config = yaml.load(stream, Loader=yaml.FullLoader)
    reconstr_type = reconstr_config["reconstruction_type"]
    reconstr_config[reconstr_type]["rank"] = peft_config_dict[adapter_name].r
    return reconstr_config, reconstr_type
