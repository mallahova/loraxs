#!/bin/bash
# export WANDB_RUN_ID="l25vss04"
# export WANDB_RESUME=true
# export PYTHONPATH=$(pwd) # should be LoRa-XS folder
export WANDB_PROJECT="rank_masking" #project folder name
# export WANDB_NOTES="Smaller learning rate, more regularization." # run description
python scripts/run_glue_rank_masking.py --target_task cola --wandb_disabled False --seed 0
# python my_experiments/tasks/task1/from_scratch/run_glue.py --target_task cola --wandb_disabled False --wandb_name run_name
