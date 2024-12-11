import argparse
import os


def glue_main(args):
    epoch = 120
    task = args.target_task  # should be one of COLA, SST2 and QNLI tasks
    model_name = "roberta-large"

    for rank in [25]:
        results_dir = f'results_{task}_{rank}'
        for lr in [1e-3]:
            for cls_lr in [5e-3]:
                for seed in [0]:
                    run_str = f'''CUDA_VISIBLE_DEVICES="0" \
                       WANDB_DISABLED="{args.wandb_diasabled}" \
                       python my_experiments/tasks/task1/from_scratch/rank_dropout.py \
                         --model_name_or_path {model_name} \
                         --lora_rank {rank} \
                         --task_name {task} \
                         --do_train \
                         --do_eval \
                         --seed {seed}\
                         --max_seq_length 128 \
                         --per_device_train_batch_size 12 \
                         --learning_rate {lr} \
                         --cls_learning_rate {cls_lr} \
                         --num_train_epochs {epoch} \
                         --save_steps 4000 \
                         --evaluation_strategy epoch  \
                         --logging_steps 20 \
                         --overwrite_output_dir \
                         --output_dir {results_dir}\
                         --k_min 10 \
                         --k_max 25 '''
                    os.system(run_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_task', required=True)
    parser.add_argument('--wandb_diasabled', required=False, default=True)
    args = parser.parse_args()

    glue_main(args)
