#!/bin/bash
# export WANDB_RUN_ID="l25vss04"
# export WANDB_RESUME=true
# export PYTHONPATH=$(pwd) # should be LoRa-XS folder
export WANDB_PROJECT="adaptive_rank_allocation" #project folder name
# export WANDB_NOTES="Smaller learning rate, more regularization." # run description
python scripts/run_glue_adaptive.py --target_task cola --wandb_disabled True --seed 0
