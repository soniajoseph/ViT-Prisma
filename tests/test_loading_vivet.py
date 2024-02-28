import pytest
import torch
from vit_prisma.models.base_vit import HookedViT
import numpy as np
import torch

from transformers import VivitForVideoClassification



def test_loading_vivet():
    TOLERANCE = 1e-4

    model_name = "google/vivit-b-16x2-kinetics400"
    batch_size = 5
    channels = 3
    num_frames = 32
    height = 224
    width = 224
    device = "cpu"

    hooked_model = HookedViT.from_pretrained(model_name, is_timm=False)
    hooked_model.to(device)
    google_model = VivitForVideoClassification.from_pretrained(model_name)
    google_model.to(device)

    with torch.random.fork_rng():
        torch.manual_seed(1)
        input_image = torch.rand((batch_size, num_frames, channels, height, width)).to(device)
    with torch.no_grad():
        hooked_output, timm_output = hooked_model(input_image), google_model(input_image).logits

    assert torch.allclose(hooked_output, timm_output, atol=TOLERANCE), f"Model output diverges! Max diff: {torch.max(torch.abs(hooked_output - timm_output))}"


