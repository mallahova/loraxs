import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import transformers
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict
from peft import PeftMixedModel, PeftModel
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import DataCollatorWithPadding, EvalPrediction, PreTrainedTokenizerBase

from utils.adaptive.rank_masking_trainer import RankMaskingTrainer

logger = logging.getLogger(__name__)


def run_prediction(
    data_args,
    is_regression: int,
    label_list: bool,
    predict_dataset: Union[Dataset, Any],
    raw_datasets: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset],
    trainer: RankMaskingTrainer,
    training_args,
):
    logger.info("*** Predict ***")

    # Loop to handle MNLI double evaluation (matched, mis-matched)
    tasks = [data_args.task_name]
    predict_datasets = [predict_dataset]
    if data_args.task_name == "mnli":
        tasks.append("mnli-mm")
        predict_datasets.append(raw_datasets["test_mismatched"])

    for predict_dataset_cur, task in zip(predict_datasets, tasks):
        # Removing the `label` columns because it contains -1 and Trainer won't like that.
        dataset_without_label = predict_dataset_cur.remove_columns("label")
        predictions = trainer.predict(dataset_without_label, metric_key_prefix="predict").predictions
        predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)

        output_predict_file = os.path.join(training_args.output_dir, f"predict_results_{task}.txt")
        if trainer.is_world_process_zero():
            with open(output_predict_file, "w") as writer:
                logger.info(f"***** Predict results {task} *****")
                writer.write("index\tprediction\n")
                for index, item in enumerate(predictions):
                    if is_regression:
                        writer.write(f"{index}\t{item:3.3f}\n")
                    else:
                        label = label_list[item]
                        writer.write(f"{index}\t{label}\n")


def run_evaluation(
    data_args,
    eval_dataset: Union[Dataset, Any],
    raw_datasets: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset],
    trainer: RankMaskingTrainer,
):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    tasks = [data_args.task_name]
    eval_datasets = [eval_dataset]
    if data_args.task_name == "mnli":
        tasks.append("mnli-mm")
        valid_mm_dataset = raw_datasets["validation_mismatched"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(valid_mm_dataset), data_args.max_eval_samples)
            valid_mm_dataset = valid_mm_dataset.select(range(max_eval_samples))
        eval_datasets.append(valid_mm_dataset)
        combined = {}

    for eval_dataset_cur, task in zip(eval_datasets, tasks):
        metrics = trainer.evaluate(eval_dataset=eval_dataset_cur)

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset_cur))

        if task == "mnli-mm":
            metrics = {k + "_mm": v for k, v in metrics.items()}
        if task is not None and "mnli" in task:
            combined.update(metrics)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", combined if task is not None and "mnli" in task else metrics)


def run_training(
    data_args,
    last_checkpoint: Optional[str],
    train_dataset: Union[Dataset, Any],
    trainer: RankMaskingTrainer,
    training_args,
):
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    max_train_samples = data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
    metrics["train_samples"] = min(max_train_samples, len(train_dataset))

    trainer.save_model()  # Saves the tokenizer too for easy upload

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    # trainer.save_state()


def create_trainer(
    compute_metrics: Tuple[Callable[[EvalPrediction], Any], Any],
    data_collator: Union[Callable[[List[Any], str], Dict[str, Any]], DataCollatorWithPadding, None],
    eval_dataset: Union[Dataset, Any],
    max_train_steps,
    model: Union[PeftModel, PeftMixedModel],
    num_update_steps_per_epoch,
    optimizer: AdamW,
    rank_allocation_args,
    scheduler: Optional[LambdaLR],
    tokenizer: Union[PreTrainedTokenizerBase, Any],
    train_dataset: Union[Dataset, Any],
    training_args,
) -> RankMaskingTrainer:
    if rank_allocation_args.memory_start is None:
        rank_allocation_args.memory_start = rank_allocation_args.memory_end = model.rank_allocation_weights.shape[0] * (
            (rank_allocation_args.rank_average) ** 2
        )  # enough memory for each weight matrix to have the average rank

    trainer = RankMaskingTrainer(
        rank_allocation_args=rank_allocation_args,
        max_train_steps=max_train_steps,
        num_update_steps_per_epoch=num_update_steps_per_epoch,
        rank_min=rank_allocation_args.rank_min,
        rank_max=rank_allocation_args.rank_max,
        memory_start=rank_allocation_args.memory_start,
        memory_end=rank_allocation_args.memory_end,
        alpha_min=rank_allocation_args.alpha_min,
        alpha_max=rank_allocation_args.alpha_max,
        epochs_memory_start=rank_allocation_args.epochs_memory_start,
        epochs_memory_start_to_end=rank_allocation_args.epochs_memory_start_to_end,
        epochs_rank_discrete=rank_allocation_args.epochs_rank_discrete,
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        optimizers=(optimizer, scheduler),
    )

    for cb in trainer.callback_handler.callbacks:
        if isinstance(cb, transformers.integrations.NeptuneCallback):
            trainer.callback_handler.remove_callback(cb)
    return trainer
