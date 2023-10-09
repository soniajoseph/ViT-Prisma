import torch.nn as nn
import logging

class Attention(nn.Module):

    def __init__(self, config, logger = None):
        super().__init__()

        self.logger = logger 
        self.config = config

        hidden_dim = self.config.transformer.hidden_dim

        assert hidden_dim % self.config.transformer.num_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.head_dim = hidden_dim // self.config.transformer.num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.q_norm, self.k_norm = nn.LayerNorm(self.head_dim) if self.config.layernorm.qknorm else nn.Identity(), nn.LayerNorm(self.head_dim) if self.config.layernorm.qknorm else nn.Identity()
        self.attn_dropout = nn.Dropout(self.config.dropout.attention) if self.config.dropout.attention > 0 else nn.Identity()
        self.proj = nn.Linear(hidden_dim, hidden_dim) if self.config.transformer.attn_hidden_layer else nn.Identity()
        self.proj_dropout = nn.Dropout(self.config.dropout.proj) if self.config.dropout.proj > 0 else nn.Identity()
        
        self._log(f"Attention layer initialized with config {self.config}")

    def _log(self, msg):
        if self.logger:
            self.logger.info(msg)

    def forward(self, x):
        self._log(f"Attention input size is {x.shape}") 
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.config.transformer.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        self._log(f"Attention size after softmax is {attn.shape}") 
        attn = self.attn_dropout(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_dropout(x)
        self._log(f"Attention output size is {x.shape}") 
        return x
