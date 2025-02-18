from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import nn

from vit_prisma.configs import HookedTextTransformerConfig
from vit_prisma.models.base_transformer import HookedTransformer
from vit_prisma.models.layers.attention import Attention
from vit_prisma.models.layers.head import Head
from vit_prisma.models.layers.layer_norm import LayerNorm, LayerNormPre
from vit_prisma.models.layers.mlp import MLP
from vit_prisma.models.layers.position_embedding import PosEmbedding
from vit_prisma.models.layers.transformer_block import TransformerBlock
from vit_prisma.prisma_tools import HookPoint
from vit_prisma.prisma_tools.activation_cache import ActivationCache


def _expand_token(token, batch_size: int):
    return token.view(1, 1, -1).expand(batch_size, -1, -1)


class HookedTextTransformer(HookedTransformer):
    """Base text model."""

    def __init__(
        self,
        cfg: HookedTextTransformerConfig,
        no_causal_mask: bool = False,
        proj_type: str = "linear",
        cls_token: bool = False,
    ):
        super().__init__()

        if isinstance(cfg, Dict):
            cfg = HookedTextTransformerConfig(**cfg)
        elif isinstance(cfg, str):
            raise ValueError(
                "Please pass in a config dictionary or HookedTextTransformerConfig"
                " object. If you want to load a pretrained model, use "
                "HookedTextTransformer.from_pretrained() instead."
            )
        self.cfg = cfg

        self.num_pos = self.context_length = self.cfg.context_length

        self.token_embed = nn.Embedding(self.cfg.vocab_size, self.cfg.d_model)
        self.hook_embed = HookPoint()
        self.pad_id = 0

        # Position embeddings
        self.pos_embed = nn.Parameter(torch.empty(self.num_pos, self.cfg.d_model))
        self.hook_pos_embed = HookPoint()

        print(f"Using a cls token: {cls_token}")
        if cls_token:
            self.cls_emb = nn.Parameter(torch.empty(self.cfg.d_model))
            self.num_pos += 1
        else:
            self.cls_emb = None

        self.hook_full_embed = HookPoint()

        if self.cfg.normalization_type == "LN":
            self.ln_pre = LayerNorm(self.cfg)
        else:
            raise ValueError(
                f"Invalid normalization type: {self.cfg.normalization_type}"
            )
        self.hook_ln_pre = HookPoint()

        block = TransformerBlock

        self.blocks = nn.ModuleList(
            [block(self.cfg, block_index) for block_index in range(self.cfg.n_layers)]
        )
        if self.cfg.normalization_type == "LN":
            self.ln_final = LayerNorm(self.cfg)
        elif self.cfg.normalization_type == "LNPre":
            self.ln_final = LayerNormPre(self.cfg)
        elif self.cfg.normalization_type is None:
            self.ln_final = nn.Identity()
        else:
            raise ValueError(
                f"Invalid normalization type: {self.cfg.normalization_type}"
            )

        self.hook_ln_final = HookPoint()

        # Final classification head
        if no_causal_mask:
            self.attn_mask = None
        else:
            print(f"registering BUFFER attn_mask")
            self.register_buffer(
                "attn_mask", self.build_causal_mask(), persistent=False
            )

        self.head = Head(self.cfg)
        self.hook_post_head_pre_normalize = HookPoint()

        self.init_weights()

        # Set up HookPoints
        self.setup()

    def build_cls_mask(self, text, cast_dtype: torch.dtype):
        cls_mask = (text != self.pad_id).unsqueeze(1)
        cls_mask = F.pad(cls_mask, (1, 0, cls_mask.shape[2], 0), value=True)
        additive_mask = torch.empty(
            cls_mask.shape, dtype=cast_dtype, device=cls_mask.device
        )
        additive_mask.fill_(0)
        additive_mask.masked_fill_(~cls_mask, float("-inf"))
        additive_mask = torch.repeat_interleave(additive_mask, self.cfg.n_heads, 0)
        return additive_mask

    def forward(
        self,
        input: Union[Float[torch.Tensor, "batch height width channels"],],
        attn_mask: Optional[Float[torch.Tensor, "batch pos pos"]] = None,
    ):
        seq_len = input.shape[1]
        token_embed = self.hook_embed(self.token_embed(input))

        attn_mask = self.attn_mask

        if self.cls_emb:
            seq_len += 1
            token_embed = torch.cat([token_embed, _expand_token(self.cls_emb, token_embed.shape[0])], dim=1)
            cls_mask = self.build_cls_mask(input, self.cfg.dtype)
            if attn_mask is not None:
                attn_mask = (
                    attn_mask[None, :seq_len, :seq_len]
                    + cls_mask[:, :seq_len, :seq_len]
                )

        pos_embed = self.hook_pos_embed(self.pos_embed[:seq_len])

        x = token_embed + pos_embed

        self.hook_full_embed(x)

        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)

        x = self.ln_final(x)
        self.hook_ln_final(x)

        x, tokens = x[torch.arange(x.shape[0]), input.argmax(dim=-1)], x

        x = x if self.cfg.return_type == "pre_logits" else self.head(x)

        self.hook_post_head_pre_normalize(x)

        if self.cfg.normalize_output:
            x = F.normalize(x, dim=-1)

        return x

    def run_with_cache(
        self, *model_args, return_cache_object=True, remove_batch_dim=False, **kwargs
    ) -> Tuple[
        Union[
            None,
            Float[torch.Tensor, "batch n_classes"],
        ],
        Union[ActivationCache, Dict[str, torch.Tensor]],
    ]:
        """Wrapper around `run_with_cache` in HookedRootModule.

        If return_cache_object is True, this will return an ActivationCache object, with a bunch of
        useful HookedTransformer specific methods, otherwise it will return a dictionary of
        activations as in HookedRootModule.
        """
        out, cache_dict = super().run_with_cache(
            *model_args, remove_batch_dim=remove_batch_dim, **kwargs
        )
        if return_cache_object:
            cache = ActivationCache(
                cache_dict, self, has_batch_dim=not remove_batch_dim
            )
            return out, cache
        else:
            return out, cache_dict

    def build_causal_mask(self):
        # lazily create causal attention mask, with full attention between the tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.num_pos, self.num_pos)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    def init_weights(self):
        if self.cls_emb:
            nn.init.normal_(self.cls_emb, std=self.cfg.cls_std)

        nn.init.normal_(self.token_embed.weight, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.01)

        if self.cfg.weight_type == "he":
            for m in self.modules():
                if isinstance(m, PosEmbedding):
                    nn.init.normal_(m.W_pos, std=self.cfg.pos_std)
                elif isinstance(m, Attention):
                    nn.init.xavier_uniform_(m.W_Q)
                    nn.init.xavier_uniform_(m.W_K)
                    nn.init.xavier_uniform_(m.W_V)
                    nn.init.xavier_uniform_(m.W_O)
                elif isinstance(m, MLP):
                    nn.init.kaiming_normal_(m.W_in, nonlinearity="relu")
                    nn.init.kaiming_normal_(m.W_out, nonlinearity="relu")
                    nn.init.zeros_(m.b_out)
                    nn.init.zeros_(m.b_in)
                elif isinstance(m, Head):
                    nn.init.kaiming_normal_(m.W_H, nonlinearity="relu")
                    nn.init.zeros_(m.b_H)
                elif isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
