#!/bin/bash
#SBATCH --job-name=adaptive_loraxs
#SBATCH --partition=dgx
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --cpus-per-task=10


for rank_allocation_learning_rate in 0.002 0.005 0.01; do
  export WANDB_PROJECT="adaptive_rank_allocation"
  export CUDA_VISIBLE_DEVICES=0  # Assign GPU 0
  export WANDB_NOTES="rank_allocation_weights initialized to zero, same scheduling, rank_max 50, average rank 20, discrete rank on the last epoch" # run description
  python scripts/run_glue_adaptive.py --target_task cola --wandb_disabled False --seed 3 --rank_allocation_learning_rate $rank_allocation_learning_rate --rank_max 50 --rank_average 20 --epoch 51 &
done

wait
