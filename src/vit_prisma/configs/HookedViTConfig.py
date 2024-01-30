from dataclasses import dataclass
import torch.nn as nn
import torch

from typing import Any, Dict, List, Optional


@dataclass
class HookedViTConfig:

    n_layers: int
    d_model: int
    d_head: int
    model_name: str = "custom"
    n_heads: int = 4
    d_mlp: Optional[int] = None
    activation_name: Optional[str] = None
    d_vocab: int = -1
    eps: float = 1e-5
    use_attn_result: bool = False
    use_attn_scale: bool = True
    use_split_qkv_input: bool = False
    use_hook_mlp_in: bool = False
    use_attn_in: bool = False
    use_local_attn: bool = False
    original_architecture: Optional[str] = None
    from_checkpoint: bool = False
    checkpoint_index: Optional[int] = None
    checkpoint_label_type: Optional[str] = None
    checkpoint_value: Optional[int] = None
    tokenizer_name: Optional[str] = None
    window_size: Optional[int] = None
    attn_types: Optional[List] = None
    init_mode: str = "gpt2"
    normalization_type: Optional[str] = "LN"
    device: Optional[str] = None
    n_devices: int = 1
    attention_dir: str = "causal"
    attn_only: bool = False
    seed: Optional[int] = None
    initializer_range: float = -1.0
    init_weights: bool = True
    scale_attn_by_inverse_layer_idx: bool = False
    positional_embedding_type: str = "standard"
    final_rms: bool = False
    d_vocab_out: int = -1
    parallel_attn_mlp: bool = False
    rotary_dim: Optional[int] = None
    n_params: Optional[int] = None
    use_hook_tokens: bool = False
    gated_mlp: bool = False
    default_prepend_bos: bool = True
    dtype: torch.dtype = torch.float32
    tokenizer_prepends_bos: Optional[bool] = None
    n_key_value_heads: Optional[int] = None
    post_embedding_ln: bool = False
    rotary_base: int = 10000
    trust_remote_code: bool = False
    rotary_adjacent_pairs: bool = False

    # Initialization
    weight_type: str = 'he'
    cls_std: float = 1e-6
    pos_std: float = 0.02

    # Image related
    n_channels: int = 3
    patch_size: int = 32
    image_size: int = 224

    # Classification related
    classification_type: str = 'cls'
    n_classes: int = 10
    return_type: str = 'pre_logits'

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        return cls(**config_dict)