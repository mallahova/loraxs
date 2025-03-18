#!/bin/bash
#SBATCH --job-name=adaptive_loraxs
#SBATCH --partition=student
#SBATCH --qos=quick
#SBATCH --gpus=1
#SBATCH --cpus-per-task=5

SEED=$1

for rank_allocation_learning_rate in 0.002 0.01 0.02; do
  export WANDB_PROJECT="adalora_rank_start_40_rank_20_random"
  export CUDA_VISIBLE_DEVICES=0  # Assign GPU 0
  export WANDB_NOTES="initial rank is 40, then it's 20 rank_allocation_weights initialized to random, same scheduling, discrete rank on the last epoch, rank_min is 5" # run description
  python scripts/run_glue_adaptive.py --target_task cola --wandb_disabled False  --seed $SEED --rank_allocation_learning_rate $rank_allocation_learning_rate --epoch 51 --rank_min 5 --rank_average 20 --rank_start 40 &
done

wait
