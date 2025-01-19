#!/bin/bash
#SBATCH --job-name=adaptive_loraxs
#SBATCH --partition=dgx
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --cpus-per-task=10


for rank_allocation_learning_rate in 0.02 0.05 0.1; do
  export WANDB_PROJECT="adaptive_rank_allocation"
  export CUDA_VISIBLE_DEVICES=0  # Assign GPU 0
  export WANDB_NOTES="rank_allocation_weights initialized to zero, same scheduling, discrete rank on the last epoch, rank_min is 5" # run description
  python scripts/run_glue_adaptive.py --target_task cola --wandb_disabled False --seed 1 --rank_allocation_learning_rate $rank_allocation_learning_rate --epoch 51 --rank_min 5 --rank_average 20 &
done

wait
