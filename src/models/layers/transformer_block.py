




import torch.nn as nn

from src.models.layers import Attention, MLP


class TransformerBlock(nn.Module):
    """
    Transformer block.
    """
    def __init__(self, config):
        super(TransformerBlock, self).__init__()
        self.config = config
        self.attention = Attention(self.config)
        self.post_attn_ln = nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps) if self.config.layer_norm_eps > 0 else nn.Identity()
        self.mlp = MLP(self.config)
        self.post_mlp_ln = nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps) if self.config.layer_norm_eps > 0 else nn.Identity()

    def forward(self, x):
        x = x + self.attention(x)
        x = self.post_attn_ln(x) 
        x = x + self.mlp(x)
        x = self.post_mlp_ln(x) 
        return x 