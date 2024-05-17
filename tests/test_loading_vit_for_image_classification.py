import pytest
import torch
from vit_prisma.models.base_vit import HookedViT
import torch

from transformers import ViTForImageClassification


def test_loading_vit_for_image_classification():
    TOLERANCE = 1e-4
    model_name="google/vit-base-patch16-224"


    batch_size = 5
    channels = 3
    height = 224
    width = 224
    device = "cpu"

    hooked_model = HookedViT.from_pretrained(model_name=model_name, is_timm=False)
    hooked_model.to(device)
    vit_model = ViTForImageClassification.from_pretrained(model_name)
    vit_model.to(device)

    with torch.random.fork_rng():
        torch.manual_seed(1)
        input_image = torch.rand((batch_size, channels, height, width)).to(device)

    
    hooked_output, vit_output = hooked_model(input_image), vit_model(input_image).logits

    assert torch.allclose(hooked_output, vit_output, atol=TOLERANCE), f"Model output diverges! Max diff: {torch.max(torch.abs(hooked_output - vit_output))}"