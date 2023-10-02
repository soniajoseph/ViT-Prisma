import torch.nn as nn

class Attention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert self.config.hidden_dim % self.config.num_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.head_dim = self.config.hidden_dim // self.config.num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(self.config.hidden_dim, 3 * self.config.hidden_dim, bias=False)
        self.q_norm, self.k_norm = nn.LayerNorm(self.head_dim) if self.config.qknorm else nn.Identity(), nn.LayerNorm(self.head_dim) if self.config.qknorm else nn.Identity()
        self.attn_dropout = nn.Dropout(attn_dropout) if self.config.attn_dropout > 0 else nn.Identity()
        self.proj = nn.Linear(self.config.hidden_dim, self.config.hidden_dim) # Do I need this layer?
        self.proj_dropout = nn.Dropout(self.config.proj_dropout) if self.config.proj_dropout > 0 else nn.Identity()

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.config.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_dropout(x)
        return x
    
    
