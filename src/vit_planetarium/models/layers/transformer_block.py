



import torch.nn as nn
from vit_planetarium.models.layers.attention import Attention
from vit_planetarium.models.layers.mlp import MLP


class TransformerBlock(nn.Module):
    """
    Transformer block.
    """
    def __init__(self, config, logger=None):
        super(TransformerBlock, self).__init__()
        self.logger = logger
        self.config = config
        layer_norm = self.config.LayerNorm.layer_norm_eps
        self.attention = Attention(self.config)
        self.post_attn_ln = nn.LayerNorm(self.config.Transformer.hidden_dim, eps=layer_norm) if layer_norm > 0 else nn.Identity()
        self.mlp = MLP(self.config)
        self.post_mlp_ln = nn.LayerNorm(self.config.Transformer.hidden_dim, eps=layer_norm) if layer_norm> 0 else nn.Identity()

    def forward(self, x):
        if self.logger:
            self.logger.info("TransformerBlock input size is {}".format(x.shape))
        x = x + self.attention(x) 
        x = self.post_attn_ln(x) 
        if self.logger:
            self.logger.info("TransformerBlock after attention size is {}".format(x.shape))
        x = x + self.mlp(x) if not self.config.Transformer.attention_only else x
        if self.logger:
            self.logger.info("TransformerBlock after MLP size is {}".format(x.shape))
        x = self.post_mlp_ln(x) if not self.config.Transformer.attention_only else x
        if self.logger:
            self.logger.info("TransformerBlock output size is {}".format(x.shape))
        return x 