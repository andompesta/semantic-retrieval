import torch
import re
from typing import List, Tuple, Optional

from semantic_retrieval.optim.adam import Adam


def get_group_params(
    named_parameters: List[Tuple[str, torch.nn.Parameter]],
    weight_decay: float,
    no_decay_patterns: Optional[List[str]] = None,
):
    """
    package the parameters in 2 groups for proper weight decay
    :param named_parameters: named parameters list
    :param weight_decay: weight decay to use
    :param no_decay_patterns: list of parameter with no decay
    :return:
    """
    optimizer_grouped_parameters = [
        dict(
            params=[
                p for n, p in named_parameters if not any(
                    [re.search(pattern, n) for pattern in no_decay_patterns]
                )
            ],
            weight_decay=weight_decay,
        ),
        dict(
            params=[
                p for n, p in named_parameters if any(
                    [re.search(pattern, n) for pattern in no_decay_patterns]
                )
            ],
            weight_decay=0.,
        )
    ]
    return optimizer_grouped_parameters


def get_optimizer(**kwargs) -> torch.optim.Optimizer:
    method = kwargs.get("method")
    params = kwargs.get("params")

    if method == "sgd":
        optim_cls = torch.optim.sgd.SGD
        optim_params = dict(
            params=params,
            lr=kwargs.get("lr", 0.001),
        )
    elif method == "rmsprop":
        optim_cls = torch.optim.rmsprop.RMSprop
        optim_params = dict(
            params=params,
            lr=kwargs.get("lr", 0.001),
            momentum=kwargs.get("momentum", 0.),
        )
    elif method == "adam":
        optim_cls = Adam
        optim_params = dict(
            params=params,
            lr=kwargs.get("lr", 0.001),
        )
    else:
        raise NotImplementedError(f"method {method} not implemented yet")

    return optim_cls(**optim_params)
