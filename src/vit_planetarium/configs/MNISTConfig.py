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
        activation_name = 'relu'
        attn_hidden_layer = False

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

    # Other Configurations
    num_classes = 10
    global_pool = False