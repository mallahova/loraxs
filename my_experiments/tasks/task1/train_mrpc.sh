#!/bin/bash
# export WANDB_RUN_ID="gmzxz8cs"
# export WANDB_RESUME=true
export PYTHONPATH=$(pwd) # should be LoRa-XS folder
# python my_experiments/tasks/task1/run_glue_pretrained.py --target_task mrpc --checkpoint_dir last --wandb_diasabled False
python my_experiments/tasks/task1/run_glue_pretrained.py --target_task mrpc --wandb_diasabled False
