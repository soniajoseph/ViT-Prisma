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
        
        layer_norm = self.config.layernorm.layer_norm_eps
        self.attention = Attention(self.config)
        self.post_attn_ln = nn.LayerNorm(self.config.transformer.hidden_dim, eps=layer_norm) if layer_norm > 0 else nn.Identity()
        self.mlp = MLP(self.config)
        self.post_mlp_ln = nn.LayerNorm(self.config.transformer.hidden_dim, eps=layer_norm) if layer_norm > 0 else nn.Identity()

    def _log(self, stage, tensor):
        if self.logger:
            self.logger.info(f"{stage} size: {tensor.shape}")

    def forward(self, x):
        self._log("TransformerBlock input", x)
        
        x = x + self.attention(x)
        x = self.post_attn_ln(x)
        self._log("TransformerBlock after attention", x)
        
        x = x + self.mlp(x) if not self.config.transformer.attention_only else x
        self._log("TransformerBlock after MLP", x)
        
        x = self.post_mlp_ln(x) if not self.config.transformer.attention_only else x
        self._log("TransformerBlock output", x)
        
        return x
