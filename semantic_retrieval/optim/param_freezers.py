from typing import List, Tuple
import torch


def unfreeze_layer_params(
    named_parameters: List[Tuple[str, torch.nn.Parameter]],
    text_layer: int,
    img_layer: int,
    unfreeze_headrs: bool = True,
):
    for n, p in named_parameters:

        if unfreeze_headrs and n in [
                "image_projection",
                "text_projection",
        ]:
            # final train projections layers
            p.requires_grad = True
            print(f"unfreeze -> {n}")

        elif n.startswith("text."):

            if n.startswith("text.resblocks."):
                layer_num = n.split(".")[2]
                if layer_num >= text_layer:
                    p.requires_grad = True
                    print(f"unfreeze -> {n}")

            if n.startswith("text.ln_final"):
                p.requires_grad = True
                print(f"unfreeze -> {n}")

        elif n.startswith("vision."):

            if n.startswith("vision.resblocks."):
                layer_num = n.split(".")[2]
                if layer_num >= img_layer:
                    p.requires_grad = True
                    print(f"unfreeze -> {n}")

            if n.startswith("vision.ln_post"):
                p.requires_grad = True
                print(f"unfreeze -> {n}")

        else:
            print(f"FREEZE -> {n}")
            p.requires_grad = False
