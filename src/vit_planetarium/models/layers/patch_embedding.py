import logging

import torch.nn as nn

class PatchEmbedding(nn.Module):

    def __init__(self, config, logger = None):
        super().__init__()
        self.logger = logger
        self.config = config
        self.proj = nn.Conv2d(self.config.n_channels, self.config.Transformer.hidden_dim, kernel_size=self.config.patch_size, stride=self.config.patch_size, bias=False)

    def forward(self, x):
        if self.logger:
            self.logger.info("PatchEmbedding input size is {}".format(x.shape))
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.logger:
            self.logger.info("PatchEmbedding output size is {}".format(x.shape))
        return x