from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class BaseClipConfig:
    emb_dim: int = 512
    # vision
    vision_emb_dim: int = 768
    vision_heads: int = 12
    vision_layers: int = 12
    img_size: tuple[int, int] = (224, 224)
    patch_size: tuple[int, int] = (16, 16)

    # text
    context_length: int = 77
    vocab_size: int = 49408
    text_emb_dim: int = 512
    text_heads: int = 8
    text_layers: int = 12

    def to_dict(self) -> Dict[str, Any]:
        return vars(self)


@dataclass
class LargeClipConfig(BaseClipConfig):
    emb_dim: int = 768
    # vision
    vision_emb_dim: int = 1024
    vision_heads: int = 16
    vision_layers: int = 24
    img_size: tuple[int, int] = (336, 336)
    patch_size: tuple[int, int] = (14, 14)

    # text
    text_emb_dim: int = 768
    text_heads: int = 12
    text_layers: int = 12
