from torch.optim.lr_scheduler import LambdaLR
from torch.optim.optimizer import Optimizer
import math


def get_constant_scheduler(
    optim: Optimizer,
    last_epoch=-1,
):
    """
    get a constant scheduler of the lr
    """
    return LambdaLR(optim, lr_lambda=lambda _: 1.0, last_epoch=last_epoch)


def get_constant_scheduler_with_warmup(
    optim: Optimizer,
    num_warmup_step: float,
    last_epoch: int = -1,
):
    """
    get a scheduler with a constant learning rate preceded by a linear warmup.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_step:
            return float(current_step) / float(max(1.0, num_warmup_step))
        return 1.0

    return LambdaLR(optim, lr_lambda, last_epoch)


def get_linear_scheduler_with_warmup(
    optim: Optimizer,
    num_warmup_step: float,
    num_training_step: int,
    last_epoch: int = -1,
):
    """
    get a scheduler that linearly increase the learning data between [0, num_warmup_steps) and then linearly decrease
    it between [num_warmup_steps, num_training_step).
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_step:
            return float(current_step) / float(max(1.0, num_warmup_step))
        return max(
            0.0,
            float(num_training_step - current_step) /
            float(max(1.0, num_training_step - num_warmup_step)),
        )

    return LambdaLR(optim, lr_lambda, last_epoch)


def get_cosine_scheduler_with_warmup(
    optim: Optimizer,
    num_warmup_step: float,
    num_training_step: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    """
    scheduler with a linear warmup between ``[0, num_warmup_step)`` and then decreases it following a cosine function
    between ``[0, pi*num_cycles]``
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_step:
            return float(current_step) / float(max(1.0, num_warmup_step))
        progress = float(current_step - num_warmup_step) / float(
            max(1, num_training_step - num_warmup_step))
        return max(
            0.0, 0.5 *
            (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optim, lr_lambda, last_epoch)


def get_cosine_with_hard_restarts_schedule_with_warmup(
    optim: Optimizer,
    num_warmup_step: float,
    num_training_step: int,
    num_cycles: float = 1.0,
    last_epoch: int = -1,
):
    """
    get a scheduler with a linear warmup between ``[0, num_warmup_step)`` and then decreases it following a cosine
    function with several hard restarts.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_step:
            return float(current_step) / float(max(1.0, num_warmup_step))
        progress = float(current_step - num_warmup_step) / float(
            max(1, num_training_step - num_warmup_step))
        if progress >= 1.0:
            return 0.0
        return max(
            0.0,
            0.5 * (1.0 + math.cos(math.pi *
                                  ((float(num_cycles) * progress) % 1.0))),
        )

    return LambdaLR(optim, lr_lambda, last_epoch)