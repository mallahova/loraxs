import json
import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import torch
from peft import LoraConfig, PeftConfig, PeftMixedModel, PeftModel
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedTokenizerBase,
)

from main_math_tuning import TrainingArguments
from utils.adaptive.args import DataTrainingArguments, ModelArguments, RankAllocaionArguments
from utils.adaptive.initialization_utils import find_and_initialize
from utils.adaptive.log import log_trainable_parameters

logger = logging.getLogger(__name__)


def initialize_model(
    adapter_name: str,
    model: Union[PeftModel, PeftMixedModel],
    peft_config_dict: Dict[str, PeftConfig],
    rank_allocation_args: RankAllocaionArguments,
    reconstr_config: Dict[str, dict],
    reconstr_type: str,
    tb_writer: SummaryWriter,
    training_args: TrainingArguments,
) -> Union[PeftModel, PeftMixedModel]:
    # Save configs
    tb_writer.add_text("peft_config_dict", str(peft_config_dict), 0)
    tb_writer.add_text("reconstr_config", str(reconstr_config), 0)
    with open(os.path.join(training_args.output_dir, "reconstr_config.json"), "w") as fp:
        json.dump(reconstr_config, fp)

    find_and_initialize(
        model,
        peft_config_dict,
        adapter_name=adapter_name,
        reconstr_type=reconstr_type,
        writer=tb_writer,
        reconstruct_config=reconstr_config,
        rank_allocation_args=rank_allocation_args,
    )

    for param in model.parameters():
        param.data = param.data.contiguous()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    log_trainable_parameters(model, tb_writer)
    return model


def configure_label_mappings(
    config: PretrainedConfig,
    data_args: DataTrainingArguments,
    is_regression: bool,
    label_list: List[str],
    model: Union[PeftModel, PeftMixedModel],
    num_labels: int,
) -> Optional[Dict[int, int]]:
    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and data_args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if sorted(label_name_to_id.keys()) == sorted(label_list):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: "
                f"model labels: {sorted(label_name_to_id.keys())}, dataset labels: {sorted(label_list)}. "
                "Ignoring the model labels as a result."
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {label_id: label for label, label_id in config.label2id.items()}
    elif data_args.task_name is not None and not is_regression:
        model.config.label2id = {label: i for i, label in enumerate(label_list)}
        model.config.id2label = {label_id: label for label, label_id in config.label2id.items()}

    return label_to_id


def load_model_and_tokenizer(
    data_args: DataTrainingArguments, model_args: ModelArguments, num_labels: int
) -> Tuple[PretrainedConfig, AutoModelForSequenceClassification, PreTrainedTokenizerBase]:
    config = AutoConfig.from_pretrained(
        (model_args.config_name if model_args.config_name else model_args.model_name_or_path),
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        (model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path),
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )
    return config, model, tokenizer


def setup_peft_config(model_args: ModelArguments) -> PeftConfig:
    return LoraConfig(
        task_type="SEQ_CLS",
        inference_mode=False,
        r=model_args.lora_rank,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=0.0,
        target_modules=["query", "value", "attention.output.dense", "output.dense"],
    )
