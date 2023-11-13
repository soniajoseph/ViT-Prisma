from dataclasses import dataclass
from vit_prisma.models.layers.transformer_block import TransformerBlock
import torch.nn as nn
from datetime import datetime

@dataclass
class ImageConfig:
    image_size: int = 64
    patch_size: int = 8
    n_channels: int = 1

@dataclass
class TransformerConfig:
    hidden_dim: int = 512
    num_heads: int = 8
    num_layers: int = 4
    block_fn = TransformerBlock
    mlp_dim: int = hidden_dim * 4  # Use a computed default
    activation_name: str = 'GELU'
    attention_only: bool = False
    attn_hidden_layer: bool = True

@dataclass
class LayerNormConfig:
    qknorm: bool = False
    layer_norm_eps: float = 0.0

@dataclass
class DropoutConfig:
    patch: float = 0.0
    position: float = 0.0
    attention: float = 0.0
    proj: float = 0.0
    mlp: float = 0.0

@dataclass
class InitializationConfig:
    weight_type: str = 'he'
    cls_std: float = 1e-6
    pos_std: float = 0.02

@dataclass
class TrainingConfig:
    loss_fn_name: str = "CrossEntropy"
    lr: float = 1e-4 
    num_epochs: int = 10
    batch_size: int = 512 # set to -1 to denote whole batch
    warmup_steps: int = 10
    weight_decay: float = 0.01
    max_grad_norm = 1.0
    device: str = 'cuda'
    seed: int = 0
    optimizer_name: str = "AdamW"
    scheduler_step: int = 200
    scheduler_gamma: float = 0.8
    early_stopping: bool = True
    early_stopping_patience: int = 3
    
@dataclass
class LoggingConfig:
    log_dir: str = 'logs'
    log_frequency: int = 100
    print_every: int = 100
    use_wandb: bool = True
    wandb_project_name = 'dsprites'

@dataclass
class SavingConfig:
    parent_dir: str = "/scratch/sjoseph/yash/dsprites"
    run_id: str = datetime.now().strftime('%Y%m%d_%H%M%S')  # Default to current timestamp, will be updated if wandb logging is on
    save_dir: str = f'checkpoints/{run_id}'  # Incorporate run_id into the save directory
    save_checkpoints: bool = True
    save_cp_frequency: int = 50

class ClassificationConfig:
    num_classes: int = 3
    global_pool: bool = False

@dataclass
class GlobalConfig:
    image: ImageConfig = ImageConfig()
    transformer: TransformerConfig = TransformerConfig()
    layernorm: LayerNormConfig = LayerNormConfig()
    dropout: DropoutConfig = DropoutConfig()
    init: InitializationConfig = InitializationConfig()
    training: TrainingConfig = TrainingConfig()
    logging: LoggingConfig = LoggingConfig()
    saving: SavingConfig = SavingConfig()
    classification: ClassificationConfig = ClassificationConfig()

