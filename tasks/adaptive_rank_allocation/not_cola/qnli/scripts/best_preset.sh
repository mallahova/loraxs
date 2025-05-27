#!/bin/bash
#SBATCH --job-name=adaptive_loraxs
#SBATCH --partition=dgx
#SBATCH --qos=quick
#SBATCH --gpus=1
#SBATCH --cpus-per-task=5

SEED=$1
export rank_allocation_lr=0.002	
export WANDB_PROJECT="adalora_qnli" #project folder name
export CUDA_VISIBLE_DEVICES=0  # Assign GPU 0
export WANDB_NOTES="rank_allocation_weights initialized to random, no alpha scheduler, rank_max is 25, average rank is 20, rank_min is 5. discrete rank on the last epoch. linear_schedule_with_warmup for lr"
python scripts/run_glue_adaptive.py --target_task qnli --wandb_disabled False  --seed $SEED \
--lr 2e-3 --cls_lr 6e-4 \
--rank_allocation_lr $rank_allocation_lr --epoch 15  --rank_min 5 --rank_max 25 --rank_average 20 --epochs_rank_discrete 1 \
--lr_scheduler linear_schedule_with_warmup \
--alpha_min 0.5 --alpha_max 3
