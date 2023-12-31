import logging
import torch.nn as nn

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
    
    def _log(self, stage, tensor):
        if self.logger:
            self.logger.info(f"{stage} size: {tensor.shape}")

    def forward(self, x):
        self._log("PatchEmbedding input", x)

        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)

        self._log("PatchEmbedding output", x)
        return x