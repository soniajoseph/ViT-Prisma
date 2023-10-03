import torch.nn as nn
import logging

class Attention(nn.Module):

    def __init__(self, config, logger = None):
        super().__init__()


        self.logger = logger 
        self.config = config

        hidden_dim = self.config.Transformer.hidden_dim


        assert hidden_dim % self.config.Transformer.num_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.head_dim = hidden_dim // self.config.Transformer.num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.q_norm, self.k_norm = nn.LayerNorm(self.head_dim) if self.config.LayerNorm.qknorm else nn.Identity(), nn.LayerNorm(self.head_dim) if self.config.LayerNorm.qknorm else nn.Identity()
        self.attn_dropout = nn.Dropout(self.config.Dropout.attention) if self.config.Dropout.attention > 0 else nn.Identity()
        self.proj = nn.Linear(hidden_dim, hidden_dim) if self.config.Transformer.attn_hidden_layer else nn.Identity()
        self.proj_dropout = nn.Dropout(self.config.Dropout.proj) if self.config.Dropout.proj > 0 else nn.Identity()
        
        if self.logger:
            self.logger.info("Attention layer initialized with config {}".format(self.config))

    def forward(self, x):
        if self.logger:
            self.logger.info("Attention input size is {}".format(x.shape)) 
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.config.Transformer.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        if self.logger:
            self.logger.info("Attention size after softmax is {}".format(attn.shape)) 
        attn = self.attn_dropout(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_dropout(x)
        if self.logger:
            self.logger.info("Attention output size is {}".format(x.shape)) 
        return x
    

