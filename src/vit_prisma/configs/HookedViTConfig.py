from dataclasses import dataclass
import torch.nn as nn
import torch

from typing import Any, Dict, List, Optional


@dataclass
class HookedViTConfig:

    n_layers: int = None
    d_model: int = None
    d_head: int = None
    d_mlp: int = None
    model_name: str = "custom"
    use_cls_token: bool = True # Off for ViT
    n_heads: int = 4
    activation_name: str = "gelu"
    d_vocab: int = -1
    eps: float = 1e-6
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
    normalize_output: bool = False
    device: Optional[str] = 'cpu'
    n_devices: int = 1
    attention_dir: str = "bidirectional"
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
        
    # Layer norm
    layer_norm_pre: bool = False # add layernorm before transformer blocks

    #Bert Block
    use_bert_block: bool = False 

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

    # Logging related
    log_dir: str = 'logs'
    use_wandb: bool = True
    wandb_team_name: str = 'perceptual-alignment'
    wandb_project_name: str = None
    log_frequency: int = 1
    print_every: int = 0


    # Training related
    optimizer_name: str = "AdamW"
    lr: float = 3e-4
    weight_decay: float = 0.01
    loss_fn_name: str = "CrossEntropy"
    batch_size: int = 512
    warmup_steps: int = 10
    scheduler_step: int = 200
    scheduler_gamma: float = .8
    scheduler_type: str = "WarmupThenStep"
    early_stopping: bool = False
    early_stopping_patience: int = 2
    num_epochs: int = 50
    max_grad_norm = 1.0
    attn_dropout_rate: float = 0.0
    mlp_dropout_rate: float = 0.0

    # Saving related
    parent_dir: str = ""
    save_dir: str = 'Checkpoints'
    save_checkpoints: bool = True
    save_cp_frequency: int = 5

    # Properties specific to video transformers
    is_video_transformer: bool = False
    video_tubelet_depth: Optional[int] = None  # Can be int or None
    video_num_frames: Optional[int] = None  # Can be int or None


    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        return cls(**config_dict)