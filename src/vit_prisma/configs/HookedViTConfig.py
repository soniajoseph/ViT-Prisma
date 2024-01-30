from dataclasses import dataclass
import torch.nn as nn
from datetime import datetime

@dataclass
class HookedViTConfig:

    # Image arguments
    image_size: int = 64
    patch_size: int = 8
    n_channels: int = 1

    # Transformer arguments
    d_model: int = 512
    num_heads: int = 8
    num_layers: int = 4
    mlp_dim: int = d_model * 4  # Use a computed default
    activation_name: str = 'GELU'
    attention_only: bool = False
    attn_hidden_layer: bool = True

    # Layernorm arguments
    final_layer_norm = True # check this 
    qknorm: bool = False
    layer_norm_eps: float = 0.0

    # Dropout arguments
    patch: float = 0.0
    position: float = 0.0
    attention: float = 0.0
    proj: float = 0.0
    mlp: float = 0.0

    # Initialization arguments
    weight_type: str = 'he'
    cls_std: float = 1e-6
    pos_std: float = 0.02

    # Classification arguments
    num_classes: int = 3
    include_cls: bool = True

