from argparse import Namespace


def compute_warmup_steps(
    args: Namespace,
    warmup_persentage: float = 1.5,
) -> Namespace:
    args.steps_per_epoch = int(args.batches_per_epoch /
                               args.gradient_accumulation_steps)
    args.num_warmup_steps = args.steps_per_epoch * warmup_persentage
    args.num_training_steps = int(args.steps_per_epoch * args.epochs)
    return args
