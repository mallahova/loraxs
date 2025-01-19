#!/bin/bash
#SBATCH --job-name=expname
#SBATCH --partition=dgx
#SBATCH --qos=normal
#SBATCH --gpus=2
#SBATCH --cpus-per-task=10

# Run the first two seeds on GPU 0
for rank in 4 16 25; do
  export WANDB_PROJECT="cola_replicated"
  export CUDA_VISIBLE_DEVICES=0  # Assign GPU 0
  python scripts/run_glue.py --seed 1 --target_task cola --rank $rank &
done

# Run the next two seeds on GPU 1
for rank in 8 12 20; do
  export WANDB_PROJECT="cola_replicated"
  export CUDA_VISIBLE_DEVICES=1  # Assign GPU 1
  python scripts/run_glue.py --seed 1 --target_task cola --rank $rank &
done

wait