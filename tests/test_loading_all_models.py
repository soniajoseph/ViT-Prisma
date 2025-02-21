import pytest
import torch
from transformers import CLIPModel, ViTModel
from vit_prisma.models.base_vit import HookedViT
import open_clip

from vit_prisma.models.model_loader import load_hooked_model
from vit_prisma.models.base_vit import HookedViT


# Define a list of models to test
MODEL_LIST = [
    


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
    "open-clip:laion/CLIP-ViT-L-14-CommonPool.XL-s13B-b90K",
    "open-clip:laion/CLIP-ViT-L-14-CommonPool.XL.clip-s13B-b90K",
    "open-clip:laion/CLIP-ViT-L-14-CommonPool.XL.laion-s13B-b90K",
    "open-clip:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K",

    "open-clip:laion/CLIP-ViT-L-14-laion2B-s32B-b82K",

    "open-clip:timm/vit_large_patch14_clip_224.laion400m_e31",
    "open-clip:timm/vit_large_patch14_clip_224.laion400m_e32",

    "open-clip:laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    "open-clip:laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",  

    "facebook/dino-vitb16",
    "facebook/dino-vitb8",

    # # MODELS THAT FAIL CURRENTLY
    "open-clip:timm/vit_medium_patch16_clip_224.tinyclip_yfcc15m",
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
    "openai/clip-vit-base-patch32", # f16, f32 issues? 


    "facebook/dino-vits16",
    "facebook/dino-vits8",


    
]

TEST_LEGACY = True
TOLERANCE = 1e-4
DEVICE = "cuda"

def pytest_configure(config):
    config.addinivalue_line(
        "markers", "divergence_info: mark test to show divergence information"
    )

# Modify your test function
@pytest.mark.divergence_info
@pytest.mark.parametrize("model_name", MODEL_LIST)
def test_loading_clip(model_name):
    """Test that the outputs of the HookedViT model match the original model."""
    input_image = generate_random_input(batch_size=5, channels=3, height=224, width=224, device=DEVICE)

    print(f"\n{'='*80}\nTesting model: {model_name}\n{'='*80}")

    try:
        if model_name.startswith("open-clip:"):
            test_open_clip_model(model_name, input_image)
        elif 'dino' in model_name:
            test_dino_model(model_name, input_image)
        else:
            test_hf_clip_model(model_name, input_image)
        print(f"\n✓ {model_name} PASSED")
    except AssertionError as e:
        print(f"\n✗ {model_name} FAILED")
        raise e

def print_divergence_info(og_output, hooked_output, model_name):
    """Print detailed divergence information between outputs."""
    diff = torch.abs(hooked_output - og_output)
    print(f"Max diff: {torch.max(diff)}")
    

def generate_random_input(batch_size, channels, height, width, device):
    """Generate a random tensor to simulate input images."""
    with torch.random.fork_rng():
        torch.manual_seed(1)
        return torch.rand((batch_size, channels, height, width)).to(device)

def test_dino_model(model_name, input_image):
    
    hf_model = ViTModel.from_pretrained(model_name)
    hf_model.to(DEVICE)
    dino_output = hf_model(input_image)
    cls_token = dino_output.last_hidden_state[:, 0]
    patches = dino_output.last_hidden_state[:, 1:]
    patches_pooled = patches.mean(dim=1)
    dino_output = torch.cat((cls_token.unsqueeze(-1), patches_pooled.unsqueeze(-1)), dim=-1)

    if TEST_LEGACY:
        hooked_model = HookedViT.from_pretrained(model_name, is_timm=False, is_clip=False, fold_ln=False)
    else:
        hooked_model = load_hooked_model(model_name)

    hooked_model.to(DEVICE)
    hooked_output = hooked_model(input_image)


    print_divergence_info(dino_output, hooked_output, model_name)
    

    # Ensure outputs are close
    assert torch.allclose(
        hooked_output, dino_output, atol=TOLERANCE
    ), f"{model_name} output diverges! Max diff: {torch.max(torch.abs(hooked_output - hf_output))}"    



def test_open_clip_model(model_name, input_image):
    """Test models from the open-clip library."""
    # Convert open-clip model name to hf-hub format
    og_model_name = "hf-hub:" + model_name[len("open-clip:"):]
    
    # Load original model
    og_model, *_ = open_clip.create_model_and_transforms(og_model_name)
    og_model.to(DEVICE)
    og_model.eval()

    # print config

    if TEST_LEGACY:
        hooked_model = HookedViT.from_pretrained(model_name, is_timm=False, is_clip=False, fold_ln=False)
    else:
        hooked_model = load_hooked_model(model_name)

    hooked_model.to(DEVICE)
    hooked_model.eval()

    print("hooked model config", hooked_model.cfg)


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



    # Load the HookedViT model
    if TEST_LEGACY:
        hooked_model = HookedViT.from_pretrained(model_name, is_timm=False, is_clip=True, fold_ln=False)

    else:
        hooked_model = load_hooked_model(model_name)

    hooked_model.to(DEVICE)
    hooked_model.eval()



    # Print confirmation of HookedViT model

    # Compare outputs
    with torch.no_grad():
        # Hugging Face CLIP vision model output
        hf_output = hf_model.get_image_features(input_image)

        # HookedViT model output
        hooked_output = hooked_model(input_image)

    # Ensure outputs are close
    assert torch.allclose(
        hooked_output, hf_output, atol=TOLERANCE
    ), f"{model_name} output diverges! Max diff: {torch.max(torch.abs(hooked_output - hf_output))}"    

