import unittest
import torch
from vit_prisma.models.layers.attention import Attention  
from vit_prisma.models.layers.mlp import MLP  
from vit_prisma.models.layers.transformer_block import TransformerBlock
from vit_prisma.models.base_vit import HookedViT
from vit_prisma.models.layers.patch_embedding import PatchEmbedding

import logging

import torch.nn as nn

from vit_prisma.configs.HookedViTConfig import HookedViTConfig


def get_test_config():
    n_layers = 1
    d_head = 8
    d_model = 8
    d_mlp=8
    return HookedViTConfig(n_layers=n_layers, d_head=d_head, d_model=d_model, d_mlp=d_mlp)

class TestAttention(unittest.TestCase):
    def test_attention(self):
        config = get_test_config()
        attention = Attention(config)
        x = torch.randn(8, 16, config.d_model)

        # Test forward pass
        output = attention(x, x, x)
        self.assertEqual(output.shape, (8, 16, config.d_model))


class TestMLP(unittest.TestCase):
    def test_mlp(self):
        config = get_test_config()
        mlp = MLP(config)
        x = torch.randn(8, 16, config.d_model)

        # Test forward pass
        output = mlp(x)
        self.assertEqual(output.shape, (8, 16, config.d_model))

class TestTransformerBlock(unittest.TestCase):
    def test_transformer_block(self):
        config = get_test_config()
        transformer_block = TransformerBlock(config)
        x = torch.randn(8, 16, config.d_model) 

        # Test forward pass
        output = transformer_block(x)
        self.assertEqual(output.shape, (8, 16, config.d_model))

class TestHookedViT(unittest.TestCase):
    def test_hooked_vit(self):
        config = get_test_config()
        config.return_type = "class_logits"
        model = HookedViT(config)
        
        x = torch.randn(8, config.n_channels, config.image_size, config.image_size)

        # Test forward pass
        output = model(x)
        self.assertEqual(output.shape, (8, config.n_classes))

class TestPatchEmbedding(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def test_patch_embedding(self):
        config = get_test_config()
        patch_embedding = PatchEmbedding(config, self.logger)
        
        x = torch.randn(8, config.n_channels, config.image_size, config.image_size)


        # Calculate expected number of patches
        num_patches = (config.image_size // config.patch_size) ** 2

        # Test forward pass
        output = patch_embedding(x)
        self.assertEqual(output.shape, (8, num_patches, config.d_model))


if __name__ == '__main__':
    unittest.main()
