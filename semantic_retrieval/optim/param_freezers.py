from typing import List, Tuple, Callable
import torch
import re


def unfreeze_layer_params(
    named_parameters: List[Tuple[str, torch.nn.Parameter]],
    layer: int = -1,
):
    if layer == -1:
        for n, p in named_parameters:
            p.requires_grad = True
            print(f"unfreeze -> {n}")
        return

    for n, p in named_parameters:
        if "embed" in n:
            p.detach_()
            p.requires_grad = False
            print(f"FREEZE -> {n}")
        elif n.startswith("roberta.encoder.layer."):
            layer_number = int(n.split(".")[3])
            if p.requires_grad and layer_number >= layer:
                p.requires_grad = True
                print(f"unfreeze -> {n}")
            else:
                p.requires_grad = False
                print(f"FREEZE -> {n}")
        elif p.requires_grad:
            p.requires_grad = True
            print(f"unfreeze -> {n}")


def unfreeze_classifier_params(
    named_parameters: List[Tuple[str, torch.nn.Parameter]]
):
    for name, param in named_parameters:
        if "xlm_roberta.encoder" in name:
            param.detach_()
            param.requires_grad = False
        elif "xlm_roberta.embeddings" in name:
            param.detach_()
            param.requires_grad = False
        elif param.requires_grad:
            param.requires_grad = True
            print("unfreeze --> {}".format(name))


def unfreeze_encoder_classifier_params(
    named_parameters: List[Tuple[str, torch.nn.Parameter]]
):
    for n, p in named_parameters:
        if "embedding" in n:
            p.detach_()
            p.requires_grad = False
        elif p.requires_grad:
            p.requires_grad = True
            print("unfreeze --> {}".format(n))


def unfreeze_all_params(named_parameters: List[Tuple[str, torch.nn.Parameter]]):
    for n, p in named_parameters:
        if "position_embeddings" in n:
            p.detach_()
            p.requires_grad = False
        elif p.requires_grad:
            p.requires_grad = True
            print("unfreeze --> {}".format(n))


def unfreeze_named_params(
    named_parameters: List[Tuple[str, torch.nn.Parameter]],
    unfreeze_name: str,
):
    for n, p in named_parameters:
        if unfreeze_name in n:
            p.requires_grad = True
            print("unfreeze --> {}".format(n))
        else:
            p.detach_()
            p.requires_grad = False


def unfreeze_layer_fn(
    template: str,
    is_distributed=False,
) -> Callable:
    if is_distributed:
        template = r"module." + template

    def unfreeze_layer(model: torch.nn.Module, unfreezing_idx: int):
        """
        function that unfreeze a given layer
        """
        unfreeze = []
        for n, p in model.named_parameters():
            if re.match(template + str(unfreezing_idx) + r"\.", n):
                unfreeze.append(n)
                p.requires_grad = True
        print(f"unfreezing block {unfreezing_idx} with {unfreeze}")

    return unfreeze_layer