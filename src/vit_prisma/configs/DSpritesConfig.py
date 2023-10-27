from dataclasses import dataclass
from vit_prisma.models.layers.transformer_block import TransformerBlock
import torch.nn as nn

@dataclass
class DatasetConfig:
    target_latent_class_idx: int = None # set to None to use all classes

@dataclass
class ImageConfig:
    image_size: int = 64
    patch_size: int = 16
    n_channels: int = 1

@dataclass
class TransformerConfig:
    hidden_dim: int = 128
    num_heads: int = 4
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
    loss_fn_name: str = "MSE"
    lr: float = 3e-4
    num_epochs: int = 50000
    batch_size: int = 64 # set to -1 to denote whole batch
    warmup_steps: int = 10
    weight_decay: float = 0.01
    max_grad_norm = 1.0
    device: str = 'mps'
    seed: int = 0
    optimizer_name: str = "AdamW"
    scheduler_step: int = 200
    scheduler_gamma: float = .8

@dataclass
class LoggingConfig:
    log_dir: str = 'logs'
    log_frequency: int = 1
    print_every: int = 0
    use_wandb: bool = True
    wandb_project_name = 'dsprites_test'

@dataclass
class SavingConfig:
    parent_dir: str = "/Users/praneets/Desktop/PRISMA/"
    save_dir: str = 'Checkpoints'
    save_checkpoints: bool = True
    save_cp_frequency: int = 10

class ClassificationConfig:
    num_classes: int = 6 if DatasetConfig.target_latent_class_idx is None else 1
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
    dataset: DatasetConfig = DatasetConfig()

