import torch
from tqdm import tqdm
import torch.nn as nn
import numpy as np

def get_activations(net,layer,data_loader,use_cuda=True, max_count=0, test_run=False, debug=True):
    '''
    Get activations of a layer for a given data_loader
    '''
    activations = []
    def hook_fn(m,i,o):
        activations.append(i[0].cpu().numpy())
    handle = layer.register_forward_hook(hook_fn)
    if use_cuda:
        net = net.cuda()
    net.eval()

    count = 0
    for i, (images, labels) in enumerate(tqdm(data_loader)):
        # if debug: print(i)
        if use_cuda:
            images = images.cuda()
        with torch.no_grad():
            output = net(images)

        count += 1
        if count > max_count:
            break
    handle.remove()
    activations_np = np.vstack(activations)     # assuming first dimension is num_examples: batches x batch_size x <feat_dims> --> num_examples x <feat_dims>
    return activations_np

class ActivationCacheHook:
    '''
    Hook to save activations of a layer
    '''
    def __init__(self):
        self.activations = []

    def __call__(self, module, input, output):
        self.activations.append(output.clone())

class CustomAttention(nn.Module):

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

        self.attn_scores = nn.Identity()
        self.attn_pattern = nn.Identity()


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
        attention_scores = self.attn_scores(attn)
        attn = attn.softmax(dim=-1)
        attn_pattern  = self.attn_pattern(attn)
        self._log(f"Attention size after softmax is {attn.shape}") 
        attn = self.attn_dropout(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_dropout(x)
        self._log(f"Attention output size is {x.shape}") 
        return x

def get_activations(net,layer,data_loader,use_cuda=True, max_count=0, test_run=False, debug=True):
    '''
    Get activations of a layer for a given data_loader
    '''
    activations = []
    def hook_fn(m,i,o):
        activations.append(i[0].cpu().numpy())
    handle = layer.register_forward_hook(hook_fn)
    if use_cuda:
        net = net.cuda()
    net.eval()

    count = 0
    for i, (images, labels) in enumerate(tqdm(data_loader)):
        # if debug: print(i)
        if use_cuda:
            images = images.cuda()
        with torch.no_grad():
            output = net(images)

        count += 1
        if count > max_count:
            break
    handle.remove()
    activations_np = np.vstack(activations)     # assuming first dimension is num_examples: batches x batch_size x <feat_dims> --> num_examples x <feat_dims>
    return activations_np


class timmCustomAttention(nn.Module):
    '''
    Custom attention layer for timm models.
    '''

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.attn_scores = nn.Identity()
        self.attn_pattern = nn.Identity()  # Initialize attn_pattern


    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale

        attn_scores = q @ k.transpose(-2, -1)
        attn_scores = self.attn_scores(attn_scores)

        attn_pattern = attn_scores.softmax(dim=-1)
        attn_pattern = self.attn_pattern(attn_pattern) # For hook function

        attn = self.attn_drop(attn_pattern)

        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x