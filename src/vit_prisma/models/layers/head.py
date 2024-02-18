
import torch.nn as nn
import torch

from typing import Dict, Union

from jaxtyping import Float

from vit_prisma.configs.HookedViTConfig import HookedViTConfig
from fancy_einsum import einsum

# maps to class logits
class Head(nn.Module):
    def __init__(self, cfg: Union[Dict, HookedViTConfig]):
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = HookedViTConfig.from_dict(cfg)
        self.cfg = cfg

        self.W_H: Float[torch.Tensor, "d_model n_classes"] = nn.Parameter(
            torch.empty(self.cfg.d_model, self.cfg.n_classes, dtype=cfg.dtype)
        )
        self.b_H: Float[torch.Tensor, "n_classes"] = nn.Parameter(
            torch.zeros(self.cfg.n_classes, dtype=cfg.dtype)
        )

    def forward(
        self, residual: Float[torch.Tensor, "batch d_model"]
    ) -> Float[torch.Tensor, "batch n_classes"]:
        
        return (
            einsum(
                "batch d_model, d_model n_classes -> batch n_classes",
                residual,
                self.W_H,
            )
            + self.b_H
        )