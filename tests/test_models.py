import unittest
import torch
from vit_planetarium.models.layers.attention import Attention  
from vit_planetarium.models.layers.mlp import MLP  
from vit_planetarium.models.layers.transformer_block import TransformerBlock
from vit_planetarium.models.base_vit import BaseViT
from vit_planetarium.models.layers.patch_embedding import PatchEmbedding

import logging

import torch.nn as nn

class Config:
    # Image and Patch Configurations
    image_size = 224
    patch_size = 16
    n_channels = 3

    # Transformer Configurations
    class Transformer:
        hidden_dim = 64
        num_heads = 4
        num_layers = 12
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
        projection = 0.0
        mlp = 0.0

    # Weight Initialization Configurations
    class Initialization:
        weight_type = 'he'
        cls_std = 1e-6
        pos_std = 0.02

    # Other Configurations
    num_classes = 10
    global_pool = False

    
    def __repr__(self):
        attributes = [f"{attr} = {getattr(self, attr)}" for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]
        return "\n".join(attributes)


class TestAttention(unittest.TestCase):

    def setUp(self):
        # Configure logging for this test case
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def test_attention(self):
        config = Config()
        attention = Attention(config, self.logger)
        x = torch.randn(8, 16, config.hidden_dim)  # Batch size: 8, Sequence length: 16

        # Test forward pass
        output = attention(x)
        self.assertEqual(output.shape, (8, 16, config.hidden_dim))

        # Test attention scores sum to 1
        with torch.no_grad():
            B, N, C = x.shape
            qkv = attention.qkv(x).reshape(B, N, 3, config.num_heads, attention.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            q, k = attention.q_norm(q), attention.k_norm(k)
            q = q * attention.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn_sum = attn.sum(dim=-1)
            self.assertTrue(torch.allclose(attn_sum, torch.ones_like(attn_sum)))

class TestMLP(unittest.TestCase):

    def setUp(self):
        # Configure logging for this test case
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def test_mlp(self):
        config = Config()
        mlp = MLP(config, self.logger)
        x = torch.randn(8, 16, config.hidden_dim)  

        # Test forward pass
        output = mlp(x)
        self.assertEqual(output.shape, (8, 16, config.hidden_dim))

class TestTransformerBlock(unittest.TestCase):

    def setUp(self):
        # Configure logging for this test case
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def test_transformer_block(self):
        config = Config()
        transformer_block = TransformerBlock(config, self.logger)
        x = torch.randn(8, 16, config.hidden_dim)  # Batch size: 8, Sequence length: 16

        # Test forward pass
        output = transformer_block(x)
        self.assertEqual(output.shape, (8, 16, config.hidden_dim))

class TestBaseViT(unittest.TestCase):

    def setUp(self):
        # Configure logging for this test case
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def test_basevit(self):
        config = Config()
        model = BaseViT(config, self.logger)
        
        # Create a dummy batch of images
        x = torch.randn(8, config.n_channels, config.image_size, config.image_size)  # Batch size: 8

        # Test forward pass
        output = model(x)
        self.assertEqual(output.shape, (8, config.num_classes))

class TestPatchEmbedding(unittest.TestCase):


    def setUp(self):
        # Configure logging for this test case
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def test_patch_embedding(self):
        config = Config()
        patch_embedding = PatchEmbedding(config, self.logger)
        
        # Create a dummy batch of images
        x = torch.randn(8, config.n_channels, config.image_size, config.image_size)  # Batch size: 8

        # Calculate expected number of patches
        num_patches = (config.image_size // config.patch_size) ** 2

        # Test forward pass
        output = patch_embedding(x)
        self.assertEqual(output.shape, (8, num_patches, config.hidden_dim))


if __name__ == '__main__':
    unittest.main()
