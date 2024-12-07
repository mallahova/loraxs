import os
import argparse
import wandb


def glue_main(args):
    batch_size=11
    seeds=[0]
    ranks=[25]
    epoch = 120
    model_name = "roberta-large"
    task = args.target_task  # should be one of MRPC, RTE and STSB tasks

    mnli_models_path = "model_checkpoints/RoBERTa-large/MNLI"

    for seed in seeds:
        for rank in ranks:
            mnli_trained_model = os.path.join(mnli_models_path, f"rank_{rank}")
            results_dir = f'results_{task}_{rank}'
            for classifier_LR in [1e-4]:
                for learning_rate in [1e-4]:
                    run_str = f'''CUDA_VISIBLE_DEVICES="0" \
                           WANDB_DISABLED="{args.wandb_diasabled}" \
                           my_experiments/tasks/task1/pretrained/rank_dropout_pretrained.py \
                             --model_name_or_path {model_name} \
                             --lora_rank {rank} \
                             --task_name {task} \
                             --do_train \
                             --do_eval \
                             --seed {seed}\
                             --max_seq_length 128 \
                             --per_device_train_batch_size {batch_size} \
                             --learning_rate {learning_rate} \
                             --cls_lr {classifier_LR}\
                             --num_train_epochs {epoch} \
                             --save_steps 2000 \
                             --evaluation_strategy epoch  \
                             --logging_steps 20 \
                             --mnli_model_path "{mnli_trained_model}"\
                             --output_dir {results_dir} \
                             --k_min 10 \
                             --k_max 25 \
                             --checkpoint_dir 2024-12-06T11:45:03-43
                           '''   
                    os.system(run_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_task', required=True)
    parser.add_argument('--wandb_diasabled', required=False, default=True)
    args = parser.parse_args()

    glue_main(args)


#output_2024-12-04 07:06:26.365174