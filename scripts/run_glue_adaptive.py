import argparse
import os


def glue_main(args):
    epoch = 50
    task = args.target_task      # should be one of COLA, SST2 and QNLI tasks
    model_name = "roberta-large"
    seeds = [int(args.seed)] if args.seed else [0, 1, 2, 3, 4]
    wandb_disabled = args.wandb_disabled
    for rank in [args.rank_max]:
        results_dir = f"results_{task}_{args.rank_min}_{args.rank_max}_{args.alpha_min}_{args.alpha_max}_{args.seed}_{args.rank_allocation_learning_rate}"
        for lr in [1e-3]:
            for cls_lr in [5e-3]:
                for seed in seeds:
                    run_str = f"""
                       WANDB_DISABLED="{wandb_disabled}" \
                       python main_glue_adaptive.py \
                         --model_name_or_path {model_name} \
                         --lora_rank {rank} \
                         --task_name {task} \
                         --do_train \
                         --do_eval \
                         --seed {seed}\
                         --max_seq_length 128 \
                         --per_device_train_batch_size 32 \
                         --learning_rate {lr} \
                         --cls_learning_rate {cls_lr} \
                         --num_train_epochs {epoch} \
                         --save_steps 6701 \
                         --evaluation_strategy epoch  \
                         --logging_steps 1 \
                         --overwrite_output_dir \
                         --output_dir {results_dir}\
                         --rank_min {args.rank_min} \
                         --rank_max {args.rank_max} \
                         --alpha_min {args.alpha_min} \
                         --rank_allocation_learning_rate {args.rank_allocation_learning_rate}"""
                    os.system(run_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_task", required=False, default="cola")
    parser.add_argument("--rank_min", required=False, default=15)
    parser.add_argument("--rank_max", required=False, default=25)
    parser.add_argument("--alpha_min", required=False, default=0.5)
    parser.add_argument("--alpha_max", required=False, default=3)
    parser.add_argument("--rank_allocation_learning_rate", required=False, default=1e-2)
    parser.add_argument("--seed", required=False, default=None)
    parser.add_argument("--wandb_disabled", required=False, default=True)


    args = parser.parse_args()

    glue_main(args)
