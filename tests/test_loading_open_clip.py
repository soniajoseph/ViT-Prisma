import pytest
import torch
import timm
from vit_prisma.models.base_vit import HookedViT
import open_clip

#currently only vit_base_patch16_224 supported (config loading issue)
def test_loading_open_clip():
    TOLERANCE = 1e-5

    batch_size = 5
    channels = 3
    height = 224
    width = 224
    device = "cpu"


    model_name = 'hf-hub:laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K'
    og_model, *data = open_clip.create_model_and_transforms(model_name)

    hooked_model = HookedViT.from_pretrained('open-clip:laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K', is_timm=False, is_clip=True, fold_ln=False) # in future, do all models
    hooked_model.to(device)

    with torch.random.fork_rng():
        torch.manual_seed(1)
        input_image = torch.rand((batch_size, channels, height, width)).to(device)

    hooked_output, og_output = hooked_model(input_image), og_model(input_image)


    og_image_embedding = og_output[0]


    print(og_output.shape)

    print(f"Hooked output type: {type(hooked_output)}")
    print(f"Original output type: {type(og_output)}")


    # print shapes
    print(f"Hooked model output shape: {hooked_output.shape}")
    print(f"Original model output shape: {og_output.shape}")

    assert torch.allclose(hooked_output, og_output, atol=TOLERANCE), f"Model output diverges! Max diff: {torch.max(torch.abs(hooked_output - og_output))}"

    print(f"Test passed successfully!")
    print(f"Max difference between outputs: {max_diff}")
    print(f"Hooked model output shape: {hooked_output.shape}")
    print(f"Original model output shape: {og_output.shape}")
    print(f"Hooked model output (first few values): {hooked_output[0, :5]}")
    print(f"Original model output (first few values): {og_output[0, :5]}")


test_loading_open_clip()