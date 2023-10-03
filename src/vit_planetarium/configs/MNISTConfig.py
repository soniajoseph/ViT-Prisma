from vit_planetarium.models.layers.transformer_block import TransformerBlock
import torch.nn as nn

class Config:
    # Image and Patch Configurations
    image_size = 28
    patch_size = 7
    n_channels = 1

    # Transformer Configurations
    class Transformer:
        hidden_dim = 128
        num_heads = 4
        num_layers = 4
        block_fn = TransformerBlock
        mlp_dim = hidden_dim * 4
        activation_fn = nn.GELU
        activation_name = 'relu' # relu for initialization purposes
        attention_only = True
        attn_hidden_layer = True

    # Layer Norm Configurations
    class LayerNorm:
        qknorm = False
        layer_norm_eps = 0.0

    # Dropout Configurations
    class Dropout:
        patch = 0.0
        position = 0.0
        attention = 0.0
        proj = 0.0
        mlp = 0.0

    # Weight Initialization Configurations
    class Initialization:
        weight_type = 'he'
        cls_std = 1e-6
        pos_std = 0.02 

    class Training:
        optimizer = ...
        lr = 1e-3
        num_epochs = 5
        batch_size = 64
        warmup_steps = 0
        weight_decay = 0.0
        max_grad_norm = None
        log_frequency = 10
        save_checkpoints = True
        save_dir = 'checkpoints'
        use_wandb = False
        wandb_project_name = None
        device = 'cuda'
        

    class Logging: 
        log_dir = 'logs'
        log_frequency = 10
        print_every = 10
        use_wandb = False
    
    class Saving:
        save_dir = 'checkpoints'
        save_frequency = 10
        
    # Other Configurations
    num_classes = 10
    global_pool = False