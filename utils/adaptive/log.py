import logging
import sys

import datasets
import transformers

logger = logging.getLogger(__name__)


def log_trainable_parameters(model, tb_writer):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    non_classifier_trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        count_params = True

        if count_params:
            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
                if "classifier" not in name:
                    non_classifier_trainable_params += num_params
    print(
        f"Non-classifier trainable params: {non_classifier_trainable_params} \n"
        f"trainable params: {trainable_params} \nall params: {all_param} \n"
        f"trainable%: {100 * trainable_params / all_param}"
    )
    tb_writer.add_text("non_classifier_trainable_params", str(non_classifier_trainable_params), 0)
    tb_writer.add_text("trainable_params", str(trainable_params), 0)


def setup_logging(training_args):
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
