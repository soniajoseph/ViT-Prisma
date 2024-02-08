"""
Attention code from timm model
"""

import torch.nn as nn
import torch

import logging
from vit_prisma.prisma import HookPoint
from vit_prisma.configs.HookedViTConfig import HookedViTConfig

from typing import Union, Dict, Optional, List, Tuple

from jaxtyping import Float, Int

import numpy as np

import einops

from vit_prisma.prisma import FactoredMatrix
from fancy_einsum import einsum

import torch.nn.functional as F



class Attention(nn.Module):

    def __init__(
            self,
            cfg: Union[Dict, HookedViTConfig],
            layer_id: Optional[int] = None,
    ):
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = HookedViTConfig.from_dict(cfg)

        self.cfg = cfg

        self.qkv = nn.Linear(self.cfg.d_model, self.cfg.d_model * 3, bias=True)
        self.proj = nn.Linear(self.cfg.d_model, self.cfg.d_model)

        # # Initialize parameters
        # self.W_Q = nn.Parameter(
        #     torch.empty(
        #         self.cfg.n_heads,
        #         self.cfg.d_model,
        #         self.cfg.d_head,
        #         dtype = self.cfg.dtype
        #     )
        # )
        # self.W_K = nn.Parameter(
        #     torch.empty(
        #         self.cfg.n_heads,
        #         self.cfg.d_model,
        #         self.cfg.d_head,
        #         dtype = self.cfg.dtype
        #     )
        # )
        # self.W_V = nn.Parameter(
        #     torch.empty(
        #         self.cfg.n_heads,
        #         self.cfg.d_model,
        #         self.cfg.d_head,
        #         dtype = self.cfg.dtype
        #     )
        # )
        # self.W_O = nn.Parameter(
        #     torch.empty(
        #         self.cfg.n_heads,
        #         self.cfg.d_head,
        #         self.cfg.d_model,
        #         dtype = self.cfg.dtype
        #     )
        # )

        # # Initialize biases
        # self.b_Q = nn.Parameter(
        #     torch.zeros(self.cfg.n_heads, self.cfg.d_head, dtype=self.cfg.dtype)
        # )
        # self.b_K = nn.Parameter(
        #     torch.zeros(self.cfg.n_heads, self.cfg.d_head, dtype=self.cfg.dtype)
        # )
        # self.b_V = nn.Parameter(
        #     torch.zeros(self.cfg.n_heads, self.cfg.d_head, dtype=self.cfg.dtype)
        # )
        # self.b_O = nn.Parameter(torch.zeros(self.cfg.d_model, dtype=self.cfg.dtype))


        # Add hook points
        self.hook_q = HookPoint()  # [batch, pos, head_index, d_head]
        self.hook_k = HookPoint()  # [batch, pos, head_index, d_head]
        self.hook_v = HookPoint()  # [batch, pos, head_index, d_head]
        self.hook_z = HookPoint()  # [batch, pos, head_index, d_head]
        self.hook_attn_scores = HookPoint()  # [batch, head_index, query_pos, key_pos]
        self.hook_pattern = HookPoint()  # [batch, head_index, query_pos, key_pos]
        # self.hook_result = HookPoint()  # [batch, pos, head_index, d_model]

        self.hook_qkv = HookPoint()  # [batch, pos, 3, head_index, d_head]

        self.layer_id = layer_id

        # # attn_scale is a constant that we divide the attention scores by pre-softmax. I'm not entirely sure why it matters, but it's probably a mix of softmax not being scale invariant and numerical stability?
        # if self.cfg.use_attn_scale:
        #     self.attn_scale = np.sqrt(self.cfg.d_head)
        # else:
        #     self.attn_scale = 1.0           

        self.scale = self.cfg.d_head ** -0.5

    @property
    def OV(self) -> FactoredMatrix:
        """
        OV-Circuit, as defined in A Mathematical Framework. Because there's no non-linearity between the value vector and the output of the layer, the output is purely determined by the matrix W_OV = W_V @ W_O, and not W_V or W_O individually. (Mathematically, for a single head, output == pattern @ residual @ W_V @ W_O, see the glossary for more)

        Done in the order W_V, W_O because the paper uses left-multiplying weight matrices, and TransformerLens uses right-multiplying, sorry!

        Returns a FactoredMatrix, with left matrix W_V [head_index, d_model, d_head] and right matrix W_O [head_index, d_head, d_model] - this is a low rank factorisation of the underlying [head_index, d_model, d_model]. FactoredMatrix has helper functions to deal with these large matrices efficiently. To get the OV circuit of a head k, attn.OV[k] works.
        """
        return FactoredMatrix(self.W_V, self.W_O)

    @property
    def QK(self) -> FactoredMatrix:
        """
        QK-Circuit, as defined in A Mathematical Framework. Because there's no non-linearity in the key-query dot product, the output is purely determined by the matrix W_QK = W_Q.T @ W_K, and not W_Q or W_K individually. (Mathematically, for a single head, pattern = destination_residual.T @ W_Q.T @ W_K @ source-residual, see the glossary for more).

        Done in the order Q on the left, K on the right, because the pattern has dimensions [destination_pos, source_pos]

        Returns a FactoredMatrix, with left matrix W_Q [head_index, d_model, d_head] and right matrix W_K.T [head_index, d_head, d_model] - this is a low rank factorisation of the underlying [head_index, d_model, d_model] matrix. FactoredMatrix has helper functions to deal with these large matrices efficiently. To get the QK circuit of a head k, attn.QK[k] works.
        """
        W_K_transpose = einops.rearrange(
            self.W_K, "head_index d_model d_head -> head_index d_head d_model"
        )
        return FactoredMatrix(self.W_Q, W_K_transpose)
    
    def forward(
            self,
            x: Union[
                Float[torch.Tensor, "batch pos d_model"],
                # Float[torch.Tensor, "batch pos head_index d_model"],
            ]
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        
        # Calculate QKV
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.cfg.n_heads, self.cfg.d_head).permute(2, 0, 3, 1, 4)

        qkv = self.hook_qkv(qkv)

        q, k, v = qkv.unbind(0)

        q = self.hook_q(q)
        k = self.hook_k(k)
        v = self.hook_v(v)
        
        # q, k, v  = self.calculate_qkv_matrices(query_input, key_input, value_input)

        # Attention Scores
        q = q * self.scale
        attn_scores = q @ k.transpose(-2, -1)
        attn_scores = self.hook_attn_scores(attn_scores)

        # Attention Pattern
        attn_pattern = attn_scores.softmax(dim=-1)
        attn_pattern = torch.where(torch.isnan(attn_pattern), torch.zeros_like(attn_pattern), attn_pattern)
        attn_pattern = self.hook_pattern(attn_pattern)
        attn_pattern = attn_pattern.to(self.cfg.dtype)

        # Value matrix
        z = attn_pattern @ v
        z = self.hook_z(z)

        # Output projection layer
        z = z.transpose(1,2).reshape(B,N,C)
        output = self.proj(z)
        # output = self.hook_result(output)

        return output

        # z = self.calculate_z_scores(v, pattern)

        # if not self.cfg.use_attn_result:
        #     out = (
        #         (
        #             einsum(
        #                 "batch pos head_index d_head, \
        #                 head_index d_head d_model -> \
        #                 batch pos d_model",
        #                 z,
        #                 self.W_O,
        #             )
        #         )
        #         + self.b_O
        #     )
        # else: 
        #     # Explicitly calculate the attention result so it can be accessed by a hook.
        #     # Off by default to not eat through GPU memory.
        #     result = self.hook_result(
        #         einsum(
        #             "batch pos head_index d_head, \
        #             head_index d_head d_model -> \
        #             batch pos head_index d_model",
        #             z,
        #             self.W_O,
        #         )
        #     )
        #     out = (
        #         einops.reduce(result, "batch pos head_index d_model -> batch position d_model", "sum")
        #         + self.b_O
        #     )
        # return out

    def calculate_qkv_matrices(
            self,
            query_input: Union[
                Float[torch.Tensor, "batch pos d_model"],
                Float[torch.Tensor, "batch pos head_index d_model"],
            ],
            key_input: Union[
                Float[torch.Tensor, "batch pos d_model"],
                Float[torch.Tensor, "batch pos head_index d_model"],
            ],
            value_input: Union[
                Float[torch.Tensor, "batch pos d_model"],
                Float[torch.Tensor, "batch pos head_index d_model"],
            ]
    ) -> Tuple[
        Float[torch.Tensor, "batch pos head_index d_head"],
        Float[torch.Tensor, "batch pos head_index d_head"],
        Float[torch.Tensor, "batch pos head_index d_head"],
    ]:
        """
        Calculate the Q, K, V matrices for the attention layer. This is done by multiplying the input by the weight matrices and adding the biases.

        Returns a tuple of (Q, K, V) matrices, each of shape [batch, pos, head_index, d_head]
        """

        if self.cfg.use_split_qkv_input or self.cfg.use_attn_in:
            qkv_einops_string = "batch pos head_index d_model"
        else:
            qkv_einops_string = "batch pos d_model"

        q = einsum(
                f"{qkv_einops_string}, head_index d_model d_head \
                -> batch pos head_index d_head",
                query_input,
                self.W_Q,
            ) + self.b_Q
            # [batch, pos, head_index, d_head]
        q = self.hook_q(q.transpose(-1,-2)) # [batch, pos, d_head, head_index]; to match timm

        k = self.hook_k(
            einsum(
                f"{qkv_einops_string}, head_index d_model d_head \
                -> batch pos head_index d_head",
                key_input,
                self.W_K,
            )
            + self.b_K
        )  # [batch, pos, head_index, d_head]
        k = self.hook_k(k.transpose(-1,-2)) # [batch, pos, d_head, head_index]; to match timm

        v = self.hook_v(
            einsum(
                f"{qkv_einops_string}, head_index d_model d_head \
                -> batch pos head_index d_head",
                value_input,
                self.W_V,
            )
            + self.b_V
        )  # [batch, pos, head_index, d_head]
        v = self.hook_v(v.transpose(-1,-2))
        return q, k, v
    
    def calculate_attn_scores(
            self,
            q: Float[torch.Tensor, "batch pos head_index d_head"],
            k: Float[torch.Tensor, "batch pos head_index d_head"],
    ) -> Float[torch.Tensor, "batch head_index query_pos key_pos"]:
        """
        Calculate the attention scores for the attention layer. This is done by multiplying the Q and K matrices together, and dividing by the square root of the dimension of the key vectors.

        Returns a tensor of shape [batch, head_index, query_pos, key_pos]
        """
        q = q * self.scale
        attn_scores = einsum(
            "batch query_pos d_head head_index, batch key_pos d_head -> batch head_index query_pos key_pos",
            q,
            k,
        )
        
        return attn_scores
    
    def calculate_z_scores(
            self,
            v: Float[torch.Tensor, "batch key_pos head_index d_head"],
            pattern: Float[torch.Tensor, "batch head_index query_pos key_pos"],
    ) -> Float[torch.Tensor, "batch query_pos head_index d_head"]:
        

        z = self.hook_z(
            einsum(
                "batch key_pos head_index d_head, \
                batch head_index query_pos key_pos -> \
                batch query_pos head_index d_head",
                v,
                pattern,
            )
        )
        return z
