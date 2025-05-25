#!/bin/bash
#SBATCH --job-name=adaptive_loraxs
#SBATCH --partition=dgx
#SBATCH --qos=quick
#SBATCH --gpus=1
#SBATCH --cpus-per-task=5

SEED=$1
rank_allocation_lr=0.002	

python scripts/run_glue_adaptive.py --target_task qnli --wandb_disabled False  --seed $SEED \
--rank_allocation_lr $rank_allocation_lr --epoch 50  --rank_min 5 --rank_max 25 --rank_average 20 --epochs_rank_discrete 1 \
--lr_scheduler linear_schedule_with_warmup \
--alpha_min 0.5 --alpha_max 3
