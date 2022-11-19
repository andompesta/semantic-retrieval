# taken from https://github.com/facebookresearch/SLIP/blob/main/models.py
from collections import OrderedDict
from functools import partial
from typing import Optional, Tuple

import torch
from torch import nn


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""
    def __init__(
            self,
            emb_dim: int,
            **kwargs
    ):
        super().__init__(emb_dim, **kwargs)

    def forward(
            self,
            x: torch.Tensor,
    ):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(
            self,
            x: torch.Tensor,
    ):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_head: int,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            batch_first: bool = False
    ):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            d_model,
            n_head,
            batch_first=batch_first
        )
        self.ln_1 = norm_layer(d_model)
        self.mlp = nn.Sequential(
            OrderedDict([
                ("c_fc", nn.Linear(d_model, d_model * 4)),
                ("gelu", act_layer()),
                ("c_proj", nn.Linear(d_model * 4, d_model))
            ])
        )
        self.ln_2 = norm_layer(d_model)

    def attention(
            self,
            x: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None
    ):
        if attn_mask is not None:
            attn_mask = attn_mask.to(
                dtype=x.dtype,
                device=x.device
            )

        return self.attn(
            query=x,
            key=x,
            value=x,
            need_weights=False,
            attn_mask=attn_mask,
        )[0]

    def forward(
            self,
            x: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attention(
            self.ln_1(x),
            attn_mask=attn_mask
        )
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):
    """
    Text transformer:
        - it uses triangular attention
        - pad-token is not needed as by default is assigned to token-id = 0
    """
    def __init__(
            self,
            vocab_size: int,
            context_length: int,
            emb_dim: int,
            layers: int,
            heads: int,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.emb_dim = emb_dim
        self.layers = layers
        self.heads = heads

        self.token_embedding = nn.Embedding(
            vocab_size,
            emb_dim
        )
        self.pos_embedding = nn.Parameter(
            torch.empty(self.context_length, emb_dim)
        )

        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(
                emb_dim,
                heads,
                act_layer=QuickGELU,
                norm_layer=LayerNorm,
            ) for _ in range(layers)
        ])

        self.ln_final = LayerNorm(self.emb_dim)

    def build_attention_mask(self) -> torch.Tensor:
        tri_mask = torch.empty(self.context_length, self.context_length)
        tri_mask.fill_(float("-inf"))
        tri_mask.triu_(1)  # zero out the lower diagonal
        return tri_mask

    def forward(
            self,
            text_ids: torch.Tensor,
    ):
        x = self.token_embedding(text_ids)  # [batch_size, ctx_len, d_model]
        x = x + self.pos_embedding
        x = x.permute(1, 0, 2)  # BLD -> LBD
        attn_mask = self.build_attention_mask().to(x.device)

        for restblock in self.resblocks:
            x = restblock(
                x,
                attn_mask=attn_mask,
            )
        
        x = x.permute(1, 0, 2)  # LBD -> BLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.emb_dim]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text_ids.argmax(dim=-1)]
        return x

    def init_weights(self):
        """ ViT weight initialization, original timm impl (for reproducibility) """
        self.apply(self._init_weights)
        nn.init.normal_(self.pos_embedding, std=0.02)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.01)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)



class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(
            self,
            img_size: Tuple[int, int],
            patch_size: Tuple[int, int],
            in_chans: int,
            emb_dim: int,
            flatten: bool = True,
            bias: bool = True,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.proj = nn.Conv2d(
            in_chans,
            emb_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=bias
    )

    def forward(
            self,
            x: torch.Tensor
    ):
        B, C, H, W = x.shape
        torch._assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        torch._assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")

        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        return x

class VisionTransformer(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
    - https://arxiv.org/abs/2010.11929
    """
    def __init__(
            self,
            img_size: Tuple[int, int] = (224, 224),
            patch_size: Tuple[int, int] = (16, 16),
            in_chans: int = 3,
            emb_dim: int = 768,
            layers: int = 12,
            heads: int = 12,
            global_pool: str = 'token',
            **kwargs,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.emb_dim = emb_dim
        self.layers = layers
        self.heads = heads

        assert global_pool in ["token", "avg"]
        self.global_pool = global_pool
        self.num_prefix_tokens = 1

        norm_layer = partial(LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            emb_dim=self.emb_dim,
        )
        num_patches = self.patch_embed.num_patches
        self.emb_len = num_patches + self.num_prefix_tokens

        self.cls_token = nn.Parameter(torch.empty(1, 1, self.emb_dim))
        self.pos_embed = nn.Parameter(torch.empty(1, self.emb_len, self.emb_dim))
        self.blocks = nn.Sequential(*[
            ResidualAttentionBlock(
                self.emb_dim,
                heads,
                act_layer=nn.GELU,
                norm_layer=norm_layer,
                batch_first=True
            )
            for _ in range(layers)
        ])
        self.norm = norm_layer(emb_dim)
        # init weights
        self.init_weights()

    def forward(
            self,
            x: torch.Tensor
    ):
        x = self.patch_embed(x)
        x = torch.cat(
            (self.cls_token.expand(x.shape[0], -1, -1), x),
            dim=1
        )
        x = x + self.pos_embed

        x = self.blocks(x)
        x = self.norm(x)

        # global pooling
        x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        return x

    def init_weights(self):
        """ ViT weight initialization, original timm impl (for reproducibility) """
        self.apply(self._init_weights)
        nn.init.zeros_(self.cls_token)
        nn.init.normal_(self.pos_embed, std=0.02)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv2d):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=.02)
        