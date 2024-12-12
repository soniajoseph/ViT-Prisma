import torch
from transformers import ViTModel

from vit_prisma.models.base_vit import HookedViT


def test_loading_dino():
    TOLERANCE = 1e-4

    model_name = "facebook/dino-vitb16"
    batch_size = 5
    channels = 3
    height = 224
    width = 224
    device = "cpu"

    dino_model = ViTModel.from_pretrained(model_name)

    hooked_model = HookedViT.from_pretrained(model_name, is_timm=False, is_clip=False, fold_ln=False)
    hooked_model.to(device)

    with torch.random.fork_rng():
        torch.manual_seed(1)
        input_image = torch.rand((batch_size, channels, height, width)).to(device)
    with torch.no_grad():
        dino_output, hooked_output = dino_model(input_image), hooked_model(input_image)

    cls_token = dino_output.last_hidden_state[:, 0]
    patches = dino_output.last_hidden_state[:, 1:]
    patches_pooled = patches.mean(dim=1)
    dino_processed_output = torch.cat((cls_token.unsqueeze(-1), patches_pooled.unsqueeze(-1)), dim=-1)
    assert torch.allclose(hooked_output, dino_processed_output, atol=TOLERANCE), f"Model output diverges! Max diff: {torch.max(torch.abs(hooked_output - dino_processed_output))}"
