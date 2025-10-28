#!/usr/bin/env python
# coding=utf-8
# Code based on the HuggingFace transformers repository.
"""Finetuning the library models for sequence classification on GLUE."""

# You can also adapt this script on your own text classification task. Pointers for this are left as comments.
import logging

from peft import (
    get_peft_model,
)
from transformers import (
    DataCollatorWithPadding,
    default_data_collator,
    set_seed,
)
from transformers.utils import send_example_telemetry
from transformers.utils.versions import require_version

from metrics import get_metric_and_compute_fn
from utils.adaptive.config_utils import check_last_checkpoint, parse_arguments, setup_output_dirs
from utils.adaptive.data_utils import get_label_info, load_datasets, prepare_train_eval_datasets, preprocess_datasets
from utils.adaptive.log import setup_logging
from utils.adaptive.model_utils import (
    configure_label_mappings,
    initialize_model,
    load_model_and_tokenizer,
    setup_peft_config,
)
from utils.adaptive.optimizer_utils import setup_optimizer_and_scheduler
from utils.adaptive.training_utils import create_trainer, run_evaluation, run_prediction, run_training

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.31.0.dev0")

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/text-classification/requirements.txt",
)

logger = logging.getLogger(__name__)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    data_args, model_args, rank_allocation_args, training_args = parse_arguments()

    ### Added code
    peft_config = setup_peft_config(model_args)

    peft_config_dict, adapter_name, reconstr_type, reconstr_config, tb_writer = setup_output_dirs(
        data_args, model_args, peft_config, training_args
    )
    ###

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_glue", model_args, data_args)

    setup_logging(training_args)

    last_checkpoint = check_last_checkpoint(training_args)

    # Set seed before initializing model.
    logger.info(f"Setting seed {training_args.seed}...")
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    raw_datasets = load_datasets(data_args, model_args, training_args)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    is_regression, label_list, num_labels = get_label_info(data_args, raw_datasets)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config, model, tokenizer = load_model_and_tokenizer(data_args, model_args, num_labels)

    model = get_peft_model(model, peft_config)

    model = initialize_model(
        adapter_name,
        model,
        peft_config_dict,
        rank_allocation_args,
        reconstr_config,
        reconstr_type,
        tb_writer,
        training_args,
    )

    label_to_id = configure_label_mappings(config, data_args, is_regression, label_list, model, num_labels)

    raw_datasets = preprocess_datasets(
        data_args, model, model_args, raw_datasets, tokenizer, training_args, label_to_id
    )
    eval_dataset, predict_dataset, train_dataset = prepare_train_eval_datasets(data_args, raw_datasets, training_args)

    compute_metrics = get_metric_and_compute_fn(data_args, is_regression)

    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    max_train_steps, num_update_steps_per_epoch, optimizer, scheduler = setup_optimizer_and_scheduler(
        model, model_args, rank_allocation_args, train_dataset, training_args
    )

    trainer = create_trainer(
        compute_metrics,
        data_collator,
        eval_dataset,
        max_train_steps,
        model,
        num_update_steps_per_epoch,
        optimizer,
        rank_allocation_args,
        scheduler,
        tokenizer,
        train_dataset,
        training_args,
    )

    # Training
    if training_args.do_train:
        run_training(data_args, last_checkpoint, train_dataset, trainer, training_args)

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        run_evaluation(data_args, eval_dataset, raw_datasets, trainer)

    if training_args.do_predict:
        run_prediction(data_args, is_regression, label_list, predict_dataset, raw_datasets, trainer, training_args)

    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "text-classification",
    }
    if data_args.task_name is not None:
        kwargs["language"] = "en"
        kwargs["dataset_tags"] = "glue"
        kwargs["dataset_args"] = data_args.task_name
        kwargs["dataset"] = f"GLUE {data_args.task_name.upper()}"

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
