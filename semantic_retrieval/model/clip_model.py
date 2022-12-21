from typing import Dict

import torch
import numpy as np
from torch import nn

from semantic_retrieval.model.modules import (
    VisionTransformer,
    Transformer,
)


class CLIP(nn.Module):

    def __init__(
        self,
        # shared embedding dimension
        emb_dim: int,

        # vision
        vision_emb_dim: int,
        vision_heads: int,
        vision_layers: int,
        img_size: tuple[int, int],
        patch_size: tuple[int, int],

        # text
        context_length: int,
        vocab_size: int,
        text_emb_dim: int,
        text_heads: int,
        text_layers: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.emb_dim = emb_dim

        self.vision = VisionTransformer(
            emb_dim=vision_emb_dim,
            heads=vision_heads,
            layers=vision_layers,
            img_size=img_size,
            patch_size=patch_size,
            **kwargs,
        )
        self.text = Transformer(
            vocab_size=vocab_size,
            context_length=context_length,
            emb_dim=text_emb_dim,
            heads=text_heads,
            layers=text_layers,
            **kwargs,
        )

        self.image_projection = nn.Parameter(
            torch.empty(vision_emb_dim, emb_dim))
        self.text_projection = nn.Parameter(torch.empty(text_emb_dim, emb_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def encode_text(
        self,
        text_ids: torch.Tensor,
    ) -> torch.Tensor:
        text_emb = self.text(text_ids)
        return text_emb @ self.text_projection

    def encode_image(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        image_emb = self.vision(image)
        return image_emb @ self.image_projection

    def forward(
        self,
        text_ids: torch.Tensor,
        images: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        image_emb = self.encode_image(images)
        text_emb = self.encode_text(text_ids)

        # normalized features
        image_emb /= image_emb.norm(
            dim=1,
            keepdim=True,
        )
        text_emb /= text_emb.norm(
            dim=1,
            keepdim=True,
        )

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_emb @ text_emb.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text

    def init_weights(self):
        self.vision.init_weights()
        self.text.init_weights()

        nn.init.normal_(
            self.image_projection,
            std=self.vision.emb_dim**-0.5,
        )
        nn.init.normal_(
            self.text_projection,
            std=self.text.emb_dim**-0.5,
        )
