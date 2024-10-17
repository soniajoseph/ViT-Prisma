import os

import torch
from vit_prisma.models.base_vit import HookedViT
from vit_prisma.utils.constants import DEVICE


def load_prisma_model(cfg, checkpoint_path):
    model_function = HookedViT
    model = model_function(cfg)

    if os.path.exists(checkpoint_path):
        print(f"Loading path at: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(DEVICE))
        print([k for k in checkpoint["model_state_dict"].keys()])
        model.load_state_dict(checkpoint["model_state_dict"])
        return model.to(DEVICE)
    else:
        raise Exception(f"Checkpoint path {checkpoint_path} does not exist")
