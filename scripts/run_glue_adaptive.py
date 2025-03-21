import argparse
import os


def glue_main(args):
    task = args.target_task      # should be one of COLA, SST2 and QNLI tasks
    model_name = "roberta-large"
    seeds = [int(args.seed)] if args.seed else [0, 1, 2, 3, 4]
    wandb_disabled = args.wandb_disabled
    if args.rank_max is None:
        if args.rank_start is not None:
            args.rank_max= args.rank_start
        else:
            args.rank_max=args.rank_average
    args.rank_max=args.rank_max if args.rank_max else args.rank_average
    args.rank_average=args.rank_average if args.rank_average else (int(args.rank_max)+int(args.rank_min))//2
    for rank in [args.rank_max]:
        results_dir = f"results_{task}_{args.rank_min}_{args.rank_max}_{args.alpha_min}_{args.alpha_max}_{args.seed}_{args.rank_allocation_learning_rate}_{args.rank_average}"
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
                         --per_device_train_batch_size {args.batch_size} \
                         --learning_rate {lr} \
                         --cls_learning_rate {cls_lr} \
                         --num_train_epochs {args.epoch} \
                         --evaluation_strategy epoch  \
                         --logging_steps 10 \
                         --overwrite_output_dir \
                         --output_dir {results_dir}\
                         --rank_min {args.rank_min} \
                         --rank_max {args.rank_max} \
                         --alpha_min {args.alpha_min} \
                         --rank_allocation_learning_rate {args.rank_allocation_learning_rate} \
                         --save_steps 0 \
                         --rank_average {args.rank_average} \
                         --rank_start {args.rank_start} \

                    """
                    os.system(run_str)
                    print(run_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_task", required=False, default="cola")
    parser.add_argument("--rank_min", required=False, default=15)
    parser.add_argument("--rank_max", required=False, default=None)
    parser.add_argument("--alpha_min", required=False, default=0.5)
    parser.add_argument("--alpha_max", required=False, default=3)
    parser.add_argument("--rank_allocation_learning_rate", required=False, default=1e-2)
    parser.add_argument("--seed", required=False, default=None)
    parser.add_argument("--wandb_disabled", required=False, default=True)
    parser.add_argument("--batch_size", required=False, default=32)
    parser.add_argument("--epoch", required=False, default=50)
    parser.add_argument("--rank_average", required=False, default=None)
    parser.add_argument("--memory_size", required=False, default=None)
    parser.add_argument("--rank_start", required=False, default=None)

    args = parser.parse_args()

    glue_main(args)
