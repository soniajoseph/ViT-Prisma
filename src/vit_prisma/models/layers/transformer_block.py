import torch
import torch.nn as nn

from vit_prisma.models.layers.attention import Attention
from vit_prisma.models.layers.mlp import MLP
from vit_prisma.models.layers.layer_norm import LayerNorm, LayerNormPre

from src.vit_prisma.configs import HookedViTConfig
from vit_prisma.prisma.hook_points import HookPoint

from typing import Dict, Optional, Tuple, Union, Float


class TransformerBlock(nn.Module):
    """
    Transformer block.
    """

    def __init__(self, cfg: Union[Dict, HookedViTConfig], block_index):
        super().__init__()

        if isinstance(cfg, Dict):
            cfg = HookedViTConfig.from_dict(cfg)
        
        self.cfg = cfg

        if self.cfg.normalization_type == "LN":
            self.ln1 = LayerNorm(self.cfg)
            if not self.cfg.attn_only:
                self.ln2 = LayerNorm(self.cfg)
        elif self.cfg.normalization_type == "LNPre":
            self.ln1 = LayerNormPre(self.cfg)
            if not self.cfg.attn_only:
                self.ln2 = LayerNormPre(self.cfg)
        elif self.cfg.normalization_type is None:
            self.ln1 = nn.Identity()
            if not self.cfg.attn_only:
                self.ln2 = nn.Identity()
        else:
            raise ValueError(f"Invalid normalization type: {self.cfg.normalization_type}")
        
        self.attn  = Attention(self.cfg)

        if not self.cfg.attn_only:
            self.mlp = MLP(self.cfg)
        
        self.hook_attn_in = HookPoint()
        self.hook_q_input = HookPoint()
        self.hook_k_input = HookPoint()
        self.hook_v_input = HookPoint()
        self.hook_mlp_in = HookPoint()
        self.hook_attn_out = HookPoint()
        self.hook_mlp_out = HookPoint()
        self.hook_resid_pre = HookPoint()

        if not self.cfg.attn_only:
            self.hook_resid_mid = HookPoint()

        self.hook_resid_post = HookPoint()

    def forward(
            self,
            resid_pre: Float[torch.Tensor, "batch pos d_model"],
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        
        resid_pre = self.hook_resid_pre(resid_pre)
        attn_in = resid_pre

        if self.cfg.use_attn_in:
            attn_in = self.hook_attn_in(attn_in.clone())
        
        if self.cfg.use_split_qkv_input:
            query_input = self.hook_q_input(attn_in.clone())
            key_input = self.hook_k_input(attn_in.clone())
            value_input = self.hook_v_input(attn_in.clone())
        else:
            query_input = attn_in
            key_input = attn_in
            value_input = attn_in
        
        attn_out = self.hook_attn_out(
            self.attn(
                query_input = self.ln1(query_input),
                key_input = self.ln1(key_input),
                value_input = self.ln1(value_input),
            )
        )
        if not self.cfg.attn_only:
            resid_mid = self.hook_resid_mid(
                resid_pre + attn_out
            )
            mlp_in = (
                resid_mid
                if not self.cfg.use_hook_mlp_in
                else self.hook_mlp_in(resid_mid.clone())
            )
            normalized_resid_mid = self.ln2(mlp_in)
            mlp_out = self.hook_mlp_out(self.mlp(normalized_resid_mid))
            resid_post = self.hook_resid_post(resid_mid + mlp_out)
        else:
            resid_post = self.hook_resid_post(resid_pre + attn_out)
        
        return resid_post
