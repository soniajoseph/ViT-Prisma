import pytest
import torch
from vit_prisma.models.base_vit import HookedViT
import numpy as np
import torch

from transformers import CLIPProcessor, CLIPModel


def test_loading_clip(model_name):
    TOLERANCE = 1e-4

    batch_size = 5
    channels = 3
    height = 224
    width = 224
    device = "cpu"

    full = CLIPModel.from_pretrained(model_name)
    tinyclip = full.vision_model

    tinyclip_final_proj = full.visual_projection
    tinyclip.to(device)
    tinyclip_final_proj.to(device)

    hooked_model = HookedViT.from_pretrained(
        model_name, is_timm=False, is_clip=True, fold_ln=False
    )
    hooked_model.to(device)

    with torch.random.fork_rng():
        torch.manual_seed(1)
        input_image = torch.rand((batch_size, channels, height, width)).to(device)
    with torch.no_grad():
        tinyclip_output, hooked_output = tinyclip_final_proj(
            tinyclip(input_image)[1]
        ), hooked_model(input_image)

    assert torch.allclose(
        hooked_output, tinyclip_output, atol=TOLERANCE
    ), f"{model_name} output diverges! Max diff: {torch.max(torch.abs(hooked_output - tinyclip_output))}"
    print(f"{model_name} output matches!")


test_loading_clip("openai/clip-vit-base-patch32")
test_loading_clip("wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M")
