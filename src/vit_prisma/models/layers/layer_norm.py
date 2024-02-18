import torch.nn as nn
import torch

from typing import Dict, Optional, Tuple, Union

from jaxtyping import Float, Int

from vit_prisma.configs.HookedViTConfig import HookedViTConfig
from vit_prisma.prisma_tools.hook_point import HookPoint

class LayerNormPre(nn.Module):
    def __init__(self, cfg: Union[Dict, HookedViTConfig]):
        """LayerNormPre - the 'center and normalise' part of LayerNorm. Length is
        normally d_model, but is d_mlp for softmax. Not needed as a parameter. This
        should only be used in inference mode after folding in LayerNorm weights"""
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = HookedViTConfig.from_dict(cfg)
        self.cfg = cfg
        self.eps = self.cfg.eps

        # Adds a hook point for the normalisation scale factor
        self.hook_scale = HookPoint()  # [batch, pos]
        # Hook Normalized captures LN output - here it's a vector with std 1 and mean 0
        self.hook_normalized = HookPoint()  # [batch, pos, length]

    def forward(
        self,
        x: Union[
            Float[torch.Tensor, "batch pos d_model"],
            Float[torch.Tensor, "batch pos head_index d_model"],
        ],
    ) -> Union[
        Float[torch.Tensor, "batch pos d_model"],
        Float[torch.Tensor, "batch pos head_index d_model"],
    ]:
        if self.cfg.dtype not in [torch.float32, torch.float64]:
            x = x.to(torch.float32)

        x = x - x.mean(axis=-1, keepdim=True)  # [batch, pos, length]
        scale: Union[
            Float[torch.Tensor, "batch pos 1"],
            Float[torch.Tensor, "batch pos head_index 1"],
        ] = self.hook_scale((x.pow(2).mean(-1, keepdim=True) + self.eps).sqrt())
        return self.hook_normalized(x / scale).to(self.cfg.dtype)


class LayerNorm(nn.Module):
    def __init__(
        self, cfg: Union[Dict, HookedViTConfig], length: Optional[int] = None
    ):
        """
        LayerNorm with optional length parameter

        length (Optional[int]): If the dimension of the LayerNorm. If not provided, assumed to be d_model
        """
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = HookedViTConfig.from_dict(cfg)
        self.cfg = cfg
        self.eps = self.cfg.eps
        if length is None:
            self.length = self.cfg.d_model
        else:
            self.length = length

        self.w = nn.Parameter(torch.ones(self.length, dtype=cfg.dtype))
        self.b = nn.Parameter(torch.zeros(self.length, dtype=cfg.dtype))

        # Adds a hook point for the normalisation scale factor
        self.hook_scale = HookPoint()  # [batch, pos, 1]
        # Hook_normalized is on the LN output
        self.hook_normalized = HookPoint()  # [batch, pos, length]

    def forward(
        self,
        x: Union[
            Float[torch.Tensor, "batch pos d_model"],
            Float[torch.Tensor, "batch pos head_index d_model"],
        ],
    ) -> Union[
        Float[torch.Tensor, "batch pos d_model"],
        Float[torch.Tensor, "batch pos head_index d_model"],
    ]:
        if self.cfg.dtype not in [torch.float32, torch.float64]:
            x = x.to(torch.float32)

        x = x - x.mean(axis=-1, keepdim=True)  # [batch, pos, length]
        scale: Float[torch.Tensor, "batch pos 1"] = self.hook_scale(
            (x.pow(2).mean(-1, keepdim=True) + self.eps).sqrt()
        )
        x = x / scale  # [batch, pos, length]
        return self.hook_normalized(x * self.w + self.b).to(self.cfg.dtype)
 