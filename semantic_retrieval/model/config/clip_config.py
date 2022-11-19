from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class ClipConfig:
    emb_dim: int = 512
    # vision
    vision_emb_dim: int = 768
    vision_heads: int = 12
    vision_layers: int = 12

    # text
    context_length: int = 77
    vocab_size: int = 49408
    text_emb_dim: int = 512
    text_heads: int = 8
    text_layers: int = 12


    def to_dict(self) -> Dict[str, Any]:
        return vars(self)