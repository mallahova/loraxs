#!/bin/bash
#SBATCH --job-name=adaptive_loraxs
#SBATCH --partition=dgx
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --cpus-per-task=10


for seed in 3 4; do
  export WANDB_PROJECT="90_percent_memory_experiment"
  export CUDA_VISIBLE_DEVICES=0  # Assign GPU 0
  export WANDB_NOTES="Memory is 10% less than memory for r=25 for every parameter matrix, rank_allocation_weights initialized to randn, same scheduling, discrete rank on the last epoch, rank_min is 5" # run description
  python scripts/run_glue_adaptive.py --target_task cola --wandb_disabled False  --seed $seed --rank_allocation_learning_rate 0.002 --epoch 51 --rank_min 5 --memory_size 54000&
done

wait
