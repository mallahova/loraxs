#!/bin/bash
#SBATCH --job-name=adaptive_loraxs
#SBATCH --partition=dgx
#SBATCH --qos=quick
#SBATCH --gpus=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G


for l_r in 0.002 0.005 0.01 ; do
  export WANDB_PROJECT="adaptive_rank_allocation"
  export CUDA_VISIBLE_DEVICES=0  # Assign GPU 0
  python scripts/run_glue_adaptive.py --target_task cola --wandb_disabled False --seed 0 --rank_allocation_learning_rate $l_r &
done

wait
