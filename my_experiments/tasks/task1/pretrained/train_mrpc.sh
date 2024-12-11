#!/bin/bash
# export WANDB_RUN_ID="l25vss04"
# export WANDB_RESUME=true
export PYTHONPATH=$(pwd) # should be LoRa-XS folder
python my_experiments/tasks/task1/pretrained/run_glue_pretrained.py --target_task mrpc --wandb_diasabled False