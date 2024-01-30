import torch.nn as nn
from src.vit_prisma.prisma.hook_points import HookPoint
from vit_prisma.models.configs.inference_configs import HookedViTConfig

from typing import Dict, Optional, Tuple, Union, Float

import torch
import torch.nn.functional as F

import numpy as np

from vit_prisma.models.activation_fns import gelu_fast, gelu_new, solu

from vit_prisma.models.layers.layer_norm import LayerNorm, LayerNormPre

import fancy_einsum as einsum

class MLP(nn.Module):

    def __init__(self, cfg: Union[Dict, HookedViTConfig]):
        super().__init__()
        
        if isinstance(cfg, HookedViTConfig):
            self.cfg = cfg
        
        self.cfg = cfg

        self.W_in = nn.Parameter(
            torch.empty(self.cfg.d_model, self.cfg.d_mlp, dtype=self.cfg.dtype)
        )
        self.b_in = nn.Parameter(
            torch.empty(self.cfg.d_mlp, dtype=self.cfg.dtype)
        )
        self.W_out = nn.Parameter(
            torch.empty(self.cfg.d_mlp, self.cfg.d_model, dtype=self.cfg.dtype)
        )
        self.b_out = nn.Parameter(
            torch.empty(self.cfg.d_model, dtype=self.cfg.dtype)
        )

        self.hook_pre = HookPoint()
        self.hook_post = HookPoint()

        if self.cfg.activation_name == "relu":
                self.act_fn = F.relu
        elif self.cfg.activation_name == "gelu":
            self.act_fn = F.gelu
        elif self.cfg.activation_name == "silu":
            self.act_fn = F.silu
        elif self.cfg.activation_name == "gelu_new":
            self.act_fn = gelu_new
        elif self.cfg.activation_name == "gelu_fast":
            self.act_fn = gelu_fast
        elif self.cfg.activation_name == "solu_ln": # why does only solu have a layernorm? 
            self.act_fn = solu
            # Hook taken between activation and layer norm
            self.hook_mid = HookPoint()  # [batch, pos, d_mlp]
            if self.cfg.normalization_type == "LN":
                self.ln = LayerNorm(self.cfg, self.cfg.d_mlp)
            else:
                self.ln = LayerNormPre(self.cfg)

        else:
            raise ValueError(f"Invalid activation function name: {self.cfg.activation_name}")
    
    def forward(self, x: Float[torch.Tensor, "batch pos d_model"]
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        
        pre_act = self.hook_pre(
              einsum("batch pos d_model, d_model mlp -> batch pos d_mlp", x, self.W_in)
            + self.b_in
        )
        if not self.cfg.act_fn.endswith("_ln"):
              post_act = self.hook_post(self.act_fn(pre_act))
        else:
              mid_act = self.hook_mid(self.act_fn(pre_act))
              post_act = self.hook_post(self.ln(mid_act))
        return (
              einsum("batch pos d_mlp, d_mlp d_model -> batch pos d_model", post_act, self.W_out) 
              + self.b_out
        )
          




    # def __init__(self, config, 
    # logger=None):
    #     super().__init__()
    #     self.logger = logger
    #     self.config = config

    #     hidden_dim = self.config.transformer.hidden_dim
    #     mlp_dim = self.config.transformer.mlp_dim

    #     self.fc1 = nn.Linear(hidden_dim, mlp_dim)
    #     self.dropout1 = nn.Dropout(self.config.dropout.mlp) if self.config.dropout.mlp > 0 else nn.Identity()
    #     self.act_fn = self.config.transformer.activation_fn()
    #     self.mlp_norm = nn.LayerNorm(mlp_dim, eps=self.config.layernorm.layer_norm_eps) if self.config.layernorm.layer_norm_eps > 0 else nn.Identity()
    #     self.fc2 = nn.Linear(mlp_dim, hidden_dim)
    #     self.dropout2 = nn.Dropout(self.config.dropout.mlp) if self.config.dropout.mlp > 0 else nn.Identity()

    # def _log(self, msg, tensor):
    #     if self.logger:
    #         self.logger.info(f"{msg} size: {tensor.shape}")

    # def forward(self, x):
    #     self._log("MLP input", x)

    #     x = self.fc1(x)
    #     x = self.act_fn(x)
    #     x = self.dropout1(x)
        
    #     self._log("MLP after first FC", x)

    #     x = self.mlp_norm(x)
    #     x = self.fc2(x)
    #     x = self.dropout2(x)

    #     self._log("MLP output", x)

    #     return x
