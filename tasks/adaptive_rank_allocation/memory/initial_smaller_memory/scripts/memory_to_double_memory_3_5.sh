#!/bin/bash
#SBATCH --job-name=adaptive_loraxs
#SBATCH --partition=dgx
#SBATCH --qos=quick
#SBATCH --gpus=1
#SBATCH --cpus-per-task=5

SEED=$1

for rank_allocation_lr in 0.002 0.01 0.02; do
  export WANDB_PROJECT="adalora_memory_to_double_memory_3_5" #project folder name
  export CUDA_VISIBLE_DEVICES=0  # Assign GPU 0
  export WANDB_NOTES="initial memory is two times smaller than memory for average rank of 25 for epochs 0-2, then it grows linearly on epochs 3-7 and stabilized at memory that is average for rank of 25 on epochs 8-49.  rank_allocation_weights initialized to random, same scheduling, discrete rank on the last epoch."
  python scripts/run_glue_adaptive.py --target_task cola --wandb_disabled False  --seed $SEED --rank_allocation_lr $rank_allocation_lr --epoch 50  --rank_min 5 --rank_max 30 --memory_start 30000 --memory_end 60000 --epochs_memory_start 3 --epochs_memory_start_to_end 5 --epochs_rank_discrete 1 &
done

wait
