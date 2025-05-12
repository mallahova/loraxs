#!/bin/bash
#SBATCH --job-name=adaptive_loraxs
#SBATCH --partition=dgx
#SBATCH --qos=quick
#SBATCH --gpus=1
#SBATCH --cpus-per-task=5

SEED=$1

for rank_allocation_lr in 0.002 0.01 0.02; do
  export WANDB_PROJECT="adalora_alpha" #project folder name
  export CUDA_VISIBLE_DEVICES=0  # Assign GPU 0
  export WANDB_NOTES="rank_allocation_weights initialized to random, exponential alpha scheduler, rank_max  is 30, average rank is 20, rank_min is 5. discrete rank on the last epoch. linear_schedule_with_warmup for lr"
  python scripts/run_glue_adaptive.py --target_task cola --wandb_disabled False  --seed $SEED \
  --rank_allocation_lr $rank_allocation_lr --epoch 50  --rank_min 5 --rank_max 30 --rank_average 20 --epochs_rank_discrete 1 \
  --lr_scheduler linear_schedule_with_warmup \
  --alpha_min 0.5 --alpha_max 3 \
  --alpha_scheduler exponential &
done

wait
