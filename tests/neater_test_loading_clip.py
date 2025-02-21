import pytest
import torch
from transformers import CLIPModel
from vit_prisma.models.base_vit import HookedViT
import open_clip

from vit_prisma.models.model_loader import load_hooked_model

# Define a list of models to test
MODEL_LIST = [

    # MChecking
    
    # # MODELS THAT FAIL CURRENTLY

    "open-clip:laion/CLIP-ViT-L-14-CommonPool.XL-s13B-b90K",
    "open-clip:laion/CLIP-ViT-L-14-CommonPool.XL.clip-s13B-b90K",
    "open-clip:laion/CLIP-ViT-L-14-CommonPool.XL.laion-s13B-b90K",
    "open-clip:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K",

    "open-clip:laion/CLIP-ViT-L-14-laion2B-s32B-b82K",

    "open-clip:timm/vit_large_patch14_clip_224.laion400m_e31",
    "open-clip:timm/vit_large_patch14_clip_224.laion400m_e32",
    "open-clip:timm/vit_medium_patch16_clip_224.tinyclip_yfcc15m",

    "open-clip:laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",  



    "open-clip:timm/vit_base_patch16_clip_224.metaclip_2pt5b",
    "open-clip:timm/vit_base_patch16_clip_224.metaclip_400m",
    "open-clip:timm/vit_base_patch16_clip_224.openai",
    "open-clip:timm/vit_base_patch32_clip_224.laion400m_e31",
    "open-clip:timm/vit_base_patch32_clip_224.laion400m_e32",
    "open-clip:timm/vit_base_patch32_clip_224.metaclip_2pt5b",
    "open-clip:timm/vit_base_patch32_clip_224.metaclip_400m",
    "open-clip:timm/vit_base_patch32_clip_224.openai",
    "open-clip:laion/CLIP-ViT-B-32-256x256-DataComp-s34B-b86K",
    "open-clip:laion/CLIP-ViT-B-32-xlm-roberta-base-laion5B-s13B-b90k",
    "open-clip:laion/CLIP-ViT-B-32-roberta-base-laion2B-s12B-b32k",
    "open-clip:laion/CLIP-ViT-H-14-frozen-xlm-roberta-large-laion5B-s13B-b90k",
    "open-clip:laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    "open-clip:timm/vit_base_patch16_plus_clip_240.laion400m_e31",
    "open-clip:timm/vit_base_patch16_plus_clip_240.laion400m_e32",
    "open-clip:timm/vit_large_patch14_clip_224.metaclip_2pt5b",
    "open-clip:timm/vit_large_patch14_clip_224.metaclip_400m",
    "open-clip:timm/vit_large_patch14_clip_224.openai",
    "open-clip:timm/vit_large_patch14_clip_336.openai",
    "open-clip:timm/vit_medium_patch32_clip_224.tinyclip_laion400m",
    "open-clip:timm/vit_xsmall_patch16_clip_224.tinyclip_yfcc15m",
    "open-clip:timm/vit_betwixt_patch32_clip_224.tinyclip_laion400m",
    "open-clip:timm/vit_gigantic_patch14_clip_224.metaclip_2pt5b",
    "open-clip:timm/vit_huge_patch14_clip_224.metaclip_2pt5b",

    "openai/clip-vit-base-patch32",


    # MODELS THAT PASS
    "wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M",
    "open-clip:laion/CLIP-ViT-B-16-CommonPool.L-s1B-b8K",
    "open-clip:laion/CLIP-ViT-B-16-CommonPool.L.basic-s1B-b8K",
    "open-clip:laion/CLIP-ViT-B-16-CommonPool.L.clip-s1B-b8K",
    "open-clip:laion/CLIP-ViT-B-16-CommonPool.L.image-s1B-b8K",
    "open-clip:laion/CLIP-ViT-B-16-CommonPool.L.laion-s1B-b8K",
    "open-clip:laion/CLIP-ViT-B-16-CommonPool.L.text-s1B-b8K",

    "open-clip:laion/CLIP-ViT-B-16-DataComp.L-s1B-b8K",
    "open-clip:laion/CLIP-ViT-B-16-DataComp.XL-s13B-b90K",

    "open-clip:laion/CLIP-ViT-B-16-laion2B-s34B-b88K",
    "open-clip:laion/CLIP-ViT-B-32-CommonPool.M-s128M-b4K",
    "open-clip:laion/CLIP-ViT-B-32-CommonPool.M.basic-s128M-b4K",
    "open-clip:laion/CLIP-ViT-B-32-CommonPool.M.clip-s128M-b4K",
    "open-clip:laion/CLIP-ViT-B-32-CommonPool.M.image-s128M-b4K",
    "open-clip:laion/CLIP-ViT-B-32-CommonPool.M.laion-s128M-b4K",
    "open-clip:laion/CLIP-ViT-B-32-CommonPool.M.text-s128M-b4K",
    "open-clip:laion/CLIP-ViT-B-32-CommonPool.S-s13M-b4K",
    "open-clip:laion/CLIP-ViT-B-32-CommonPool.S.basic-s13M-b4K",
    "open-clip:laion/CLIP-ViT-B-32-CommonPool.S.clip-s13M-b4K",
    "open-clip:laion/CLIP-ViT-B-32-CommonPool.S.image-s13M-b4K",
    "open-clip:laion/CLIP-ViT-B-32-CommonPool.S.laion-s13M-b4K",
    "open-clip:laion/CLIP-ViT-B-32-CommonPool.S.text-s13M-b4K",

    "open-clip:laion/CLIP-ViT-B-32-DataComp.M-s128M-b4K",
    "open-clip:laion/CLIP-ViT-B-32-DataComp.S-s13M-b4K",
    "open-clip:laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K",

    "open-clip:laion/CLIP-ViT-B-32-laion2B-s34B-b79K",

    "open-clip:timm/vit_base_patch16_clip_224.laion400m_e31",
    "open-clip:timm/vit_base_patch16_clip_224.laion400m_e32",
    "open-clip:timm/vit_base_patch32_clip_224.laion2b_e16",
    
]

TOLERANCE = 1e-4
DEVICE = "cuda"


@pytest.mark.parametrize("model_name", MODEL_LIST)
def test_loading_clip(model_name):
    """Test that the outputs of the HookedViT model match the original model."""
    # Generate a random input image
    input_image = generate_random_input(batch_size=5, channels=3, height=224, width=224, device=DEVICE)

    print(f"Testing model: {model_name}")

    if model_name.startswith("open-clip:"):
        test_open_clip_model(model_name, input_image)
    else:
        test_hf_clip_model(model_name, input_image)

def print_divergence_info(og_output, hooked_output, model_name):
    """Print detailed divergence information between outputs."""
    diff = torch.abs(hooked_output - og_output)
    max_diff = torch.max(diff).item()  # Convert to Python scalar
    mean_diff = torch.mean(diff).item()
    median_diff = torch.median(diff).item()
    
    print(f"\nDivergence Analysis for {model_name}:")
    print(f"Max difference:     {max_diff:.8f}")
    print(f"Mean difference:    {mean_diff:.8f}")
    print(f"Median difference:  {median_diff:.8f}")
    
    # Print location of max difference
    if max_diff > 0:
        max_loc = torch.where(diff == max_diff)
        print(f"Location of max difference: {tuple(idx.tolist() for idx in max_loc)}")
        print(f"Original value at max diff: {og_output[max_loc].item():.8f}")
        print(f"Hooked value at max diff:   {hooked_output[max_loc].item():.8f}")


def generate_random_input(batch_size, channels, height, width, device):
    """Generate a random tensor to simulate input images."""
    with torch.random.fork_rng():
        torch.manual_seed(1)
        return torch.rand((batch_size, channels, height, width)).to(device)


def test_open_clip_model(model_name, input_image):
    """Test models from the open-clip library."""
    # Convert open-clip model name to hf-hub format
    og_model_name = "hf-hub:" + model_name[len("open-clip:"):]
    
    # Load original model
    og_model, *_ = open_clip.create_model_and_transforms(og_model_name)
    og_model.to(DEVICE)
    og_model.eval()


    hooked_model = load_hooked_model(model_name)
    hooked_model.to(DEVICE)
    hooked_model.eval()

    print("Model config", hooked_model.cfg)

    # Compare outputs
    with torch.no_grad():
        og_output, *_ = og_model(input_image)
        hooked_output = hooked_model(input_image)

    print_divergence_info(og_output, hooked_output, model_name)

    assert torch.allclose(
        og_output, hooked_output, atol=TOLERANCE
    ), f"{model_name} output diverges! Max diff: {torch.max(torch.abs(hooked_output - og_output))}"

def test_hf_clip_model(model_name, input_image):
    """Test models from Hugging Face's CLIP library."""
    # Load the full Hugging Face CLIP model
    hf_model = CLIPModel.from_pretrained(model_name)
    hf_model.to(DEVICE)
    hf_model.eval()

    # Print the loaded model name
    print(f"Loaded HuggingFace CLIP model: {model_name}")

    print("HF config:", hf_model.config)

    # Load the HookedViT model
    hooked_model = load_hooked_model(model_name)

    hooked_model.to(DEVICE)
    hooked_model.eval()

    print("Hooked config:", hooked_model.cfg)
    print(0)


    # Print confirmation of HookedViT model
    print(f"Loaded HookedViT model for: {model_name}")

    # Compare outputs
    with torch.no_grad():
        # Hugging Face CLIP vision model output
        hf_output = hf_model.get_image_features(input_image)

        # HookedViT model output
        hooked_output = hooked_model(input_image)

    print_divergence_info(hf_output, hooked_output, model_name)


    # Ensure outputs are close
    assert torch.allclose(
        hooked_output, hf_output, atol=TOLERANCE
    ), f"{model_name} output diverges! Max diff: {torch.max(torch.abs(hooked_output - hf_output))}"    


if __name__ == "__main__":
    for model_name in MODEL_LIST:
        print(f"Testing model: {model_name}")

        # Generate a random input image
        input_image = generate_random_input(batch_size=5, channels=3, height=224, width=224, device=DEVICE)

        if model_name.startswith("open-clip:"):
            test_open_clip_model(model_name, input_image)
        else:
            test_hf_clip_model(model_name, input_image)

