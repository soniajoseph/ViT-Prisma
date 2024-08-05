from typing import Any, cast

import torch

from vit_prisma.models.base_vit import HookedViT
from vit_prisma.prisma_tools.hooked_root_module import HookedRootModule


def load_model(
    model_class_name: str,
    model_name: str,
    device: str | torch.device | None = None,
    model_from_pretrained_kwargs: dict[str, Any] | None = None,
) -> HookedRootModule:
    model_from_pretrained_kwargs = model_from_pretrained_kwargs or {}
    
    if "n_devices" in model_from_pretrained_kwargs:
        n_devices = model_from_pretrained_kwargs["n_devices"]
        if n_devices > 1:
            print("MODEL LOADING:")
            print("Setting model device to cuda for d_devices")
            print(f"Will use cuda:0 to cuda:{n_devices-1}")
            device = "cuda"
            print("-------------")

    if model_class_name == "HookedViT":
        is_timm, is_clip = set_flags(model_name)
        return HookedViT.from_pretrained(model_name, is_timm=is_timm, is_clip=is_clip).to(device)
    
    else:  # pragma: no cover
        raise ValueError(f"Unknown model class: {model_class_name}")
    

def set_flags(model_name):
    model_name_lower = model_name.lower()
    is_clip = 'clip' in model_name_lower
    is_timm = not is_clip
    return is_timm, is_clip
