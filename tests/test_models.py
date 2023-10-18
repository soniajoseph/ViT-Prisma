import unittest
import torch
from vit_prisma.models.layers.attention import Attention  
from vit_prisma.models.layers.mlp import MLP  
from vit_prisma.models.layers.transformer_block import TransformerBlock
from vit_prisma.models.base_vit import BaseViT
from vit_prisma.models.layers.patch_embedding import PatchEmbedding

import logging

import torch.nn as nn

from vit_prisma.configs.MNISTConfig import GlobalConfig


class TestAttention(unittest.TestCase):

    def setUp(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def test_attention(self):
        config = GlobalConfig()
        attention = Attention(config, self.logger)
        x = torch.randn(8, 16, config.transformer.hidden_dim)

        # Test forward pass
        output = attention(x)
        self.assertEqual(output.shape, (8, 16, config.transformer.hidden_dim))


class TestMLP(unittest.TestCase):

    def setUp(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def test_mlp(self):
        config = GlobalConfig()
        mlp = MLP(config, self.logger)
        x = torch.randn(8, 16, config.transformer.hidden_dim)

        # Test forward pass
        output = mlp(x)
        self.assertEqual(output.shape, (8, 16, config.transformer.hidden_dim))

class TestTransformerBlock(unittest.TestCase):

    def setUp(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def test_transformer_block(self):
        config = GlobalConfig()
        transformer_block = TransformerBlock(config, self.logger)
        x = torch.randn(8, 16, config.transformer.hidden_dim) 

        # Test forward pass
        output = transformer_block(x)
        self.assertEqual(output.shape, (8, 16, config.transformer.hidden_dim))

class TestBaseViT(unittest.TestCase):

    def setUp(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def test_basevit(self):
        config = GlobalConfig()
        model = BaseViT(config, self.logger)
        
        x = torch.randn(8, config.image.n_channels, config.image.image_size, config.image.image_size)

        # Test forward pass
        output = model(x)
        self.assertEqual(output.shape, (8, config.classification.num_classes))

class TestPatchEmbedding(unittest.TestCase):

    def setUp(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def test_patch_embedding(self):
        config = GlobalConfig()
        patch_embedding = PatchEmbedding(config, self.logger)
        
        x = torch.randn(8, config.image.n_channels, config.image.image_size, config.image.image_size)


        # Calculate expected number of patches
        num_patches = (config.image.image_size // config.image.patch_size) ** 2

        # Test forward pass
        output = patch_embedding(x)
        self.assertEqual(output.shape, (8, num_patches, config.transformer.hidden_dim))


if __name__ == '__main__':
    unittest.main()
