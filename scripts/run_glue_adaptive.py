import argparse
import os


def glue_main(args):
    task = args.target_task      # should be one of COLA, SST2 and QNLI tasks
    model_name = "roberta-large"
    seeds = [int(args.seed)] if args.seed else [0, 1, 2, 3, 4]
    wandb_disabled = args.wandb_disabled
    learning_rates=[1e-3]
    cls_learning_rates=[5e-3]
    if args.rank_max is None:
        if args.rank_start is not None:
            args.rank_max= args.rank_start
        else:
            args.rank_max=args.rank_average
    for lr in learning_rates:
        for cls_lr in cls_learning_rates:
            for seed in seeds:
                results_dir = f"results_{task}_{args.rank_min}_{args.rank_max}_{args.alpha_min}_{args.alpha_max}_{seed}_{args.rank_allocation_lr}_{args.rank_average}"
                run_str = [
                    f'WANDB_DISABLED="{wandb_disabled}"',
                    "python", "main_glue_adaptive.py",
                    "--model_name_or_path", model_name,
                    "--lora_rank", str(args.rank_max),
                    "--task_name", task,
                    "--seed", str(seed),
                    "--per_device_train_batch_size", str(args.batch_size),
                    "--num_train_epochs", str(args.epoch),
                    "--output_dir", results_dir,
                    "--learning_rate", lr,
                    "--cls_learning_rate", cls_lr,
                    "--do_train", "--do_eval",
                    "--max_seq_length", "128",
                    "--evaluation_strategy", "epoch",
                    "--logging_steps", "10",
                    "--overwrite_output_dir",
                    "--save_steps", "0",
                ]

                # Helper function to add arguments conditionally
                def add_arg(arg_name, arg_value):
                    if arg_value is not None:
                        run_str.extend([f"--{arg_name}", str(arg_value)])

                add_arg("rank_average", args.rank_average)
                add_arg("rank_min", args.rank_min)
                add_arg("rank_max", args.rank_max)
                add_arg("memory_start", args.memory_start)
                add_arg("memory_end", args.memory_end)
                add_arg("epochs_memory_start", args.epochs_memory_start)
                add_arg("epochs_memory_start_to_end", args.epochs_memory_start_to_end)
                add_arg("epochs_rank_discrete", args.epochs_rank_discrete)
                add_arg("alpha_min", args.alpha_min)
                add_arg("alpha_max", args.alpha_max)
                add_arg("rank_allocation_lr", args.rank_allocation_lr)
                add_arg("lr_scheduler", args.lr_scheduler)
                os.system(" ".join(run_str))
                print(run_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_disabled", required=False, default=True)

    parser.add_argument("--target_task", required=False, default="cola")
    parser.add_argument("--batch_size", required=False, default=32)
    parser.add_argument("--epoch", required=False, default=50)

    parser.add_argument("--rank_average", required=False, default=None)

    parser.add_argument("--rank_min", required=False, default=None)
    parser.add_argument("--rank_max", required=False, default=None)

    parser.add_argument("--memory_start", required=False, default=None)
    parser.add_argument("--memory_end", required=False, default=None)

    parser.add_argument("--epochs_memory_start", required=False, default=None)
    parser.add_argument("--epochs_memory_start_to_end", required=False, default=None)
    parser.add_argument("--epochs_rank_discrete", required=False, default=None)
    
    parser.add_argument("--alpha_min", required=False, default=0.5)
    parser.add_argument("--alpha_max", required=False, default=3)

    parser.add_argument("--rank_allocation_lr", required=False, default=1e-2)
    parser.add_argument("--lr_scheduler", required=False, default="linear_schedule_with_warmup",  choices= ["linear_schedule_with_warmup", "constant_schedule"])


    parser.add_argument("--seed", required=False, default=None)


    args = parser.parse_args()
    glue_main(args)
