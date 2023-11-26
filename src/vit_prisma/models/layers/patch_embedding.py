import logging
import torch.nn as nn
import torch
from vit_prisma.training.prisma_types import Masking, Objective

class PatchEmbedding(nn.Module):

    def __init__(self, config, logger=None):
        super().__init__()
        self.logger = logger
        self.config = config
        self.proj = nn.Conv2d(
            self.config.image.n_channels, 
            self.config.transformer.hidden_dim, 
            kernel_size=self.config.image.patch_size, 
            stride=self.config.image.patch_size, 
            bias=False
        )
        self.is_masking_enabled = (config.training.objective == Objective.GENERATION)
        self.masking_type = config.mask.mask_type
        self.mask_prob = config.mask.mask_prob
    
    def _log(self, stage, tensor):
        if self.logger:
            self.logger.info(f"{stage} size: {tensor.shape}")

    def forward(self, x):
        self._log("PatchEmbedding input", x)

        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)

        if self.is_masking_enabled and self.masking_type == Masking.RANDOM:
            num_patches = x.shape[1]
            num_patches_to_mask = int(num_patches * self.mask_prob)
            idxs = torch.randperm(num_patches)[:num_patches_to_mask]
            x[:, idxs, :] = 0

        self._log("PatchEmbedding output", x)
        return x