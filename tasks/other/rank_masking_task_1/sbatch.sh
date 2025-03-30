#!/bin/bash
#SBATCH --job-name=expname
#SBATCH --partition=dgx
#SBATCH --qos=normal
#SBATCH --gpus=2
#SBATCH --cpus-per-task=10

# Run the first two seeds on GPU 0
for seed in 0 1 2; do
  export WANDB_PROJECT="rank_masking"
  export CUDA_VISIBLE_DEVICES=0  # Assign GPU 0
  python scripts/run_glue_rank_masking.py --target_task cola --wandb_disabled False --seed $seed &
done

for seed in 3 4 5; do
  export WANDB_PROJECT="rank_masking"
  export CUDA_VISIBLE_DEVICES=1  # Assign GPU 0
  python scripts/run_glue_rank_masking.py --target_task cola --wandb_disabled False --seed $seed &
done


wait