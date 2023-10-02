import unittest
import torch
from vit_planetarium.models.layers.attention import Attention  
from vit_planetarium.models.layers.mlp import MLP  
from vit_planetarium.models.layers.transformer_block import TransformerBlock
from vit_planetarium.models.base_vit import BaseViT

import logging

import torch.nn as nn

class Config:
    hidden_dim = 64
    num_heads = 4
    activation_fn = nn.GELU
    mlp_dim = hidden_dim * 4
    attn_hidden_layer = False
    attn_dropout = 0.1
    proj_dropout = 0.1
    mlp_dropout = 0.0
    layer_norm_eps = 1e-6
    qknorm = True
    weight_init_type = 'he'
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
        x = torch.randn(8, config.channels, config.image_size, config.image_size)  # Batch size: 8

        # Test forward pass
        output = model(x)
        self.assertEqual(output.shape, (8, config.num_classes))


if __name__ == '__main__':
    unittest.main()
