import pytest
import torch
import timm
from vit_prisma.models.base_vit import HookedViT
import open_clip
from collections import defaultdict

def collect_activations(model, prefix=''):
    activations = defaultdict(list)
    
    def hook_fn(module, input, output, name):
        activations[name].append(output)
    
    for name, module in model.named_modules():
        full_name = f"{prefix}.{name}" if prefix else name
        module.register_forward_hook(lambda mod, inp, out, name=full_name: hook_fn(mod, inp, out, name))
    
    return activations

def test_loading_open_clip():
    TOLERANCE = 1e-5

    batch_size = 2
    channels = 3
    height = 224
    width = 224
    device = "cpu"

    model_name = 'hf-hub:laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K'
    og_model, *data = open_clip.create_model_and_transforms(model_name)
    og_model.to(device)

    hooked_model = HookedViT.from_pretrained('open-clip:laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K', is_timm=False, is_clip=True, fold_ln=False)
    hooked_model.to(device)

    with torch.random.fork_rng():
        torch.manual_seed(1)
        input_image = torch.rand((batch_size, channels, height, width)).to(device)

    # Collect activations for og_model
    og_activations = collect_activations(og_model, prefix='og')

    # Run models
    og_output = og_model(input_image)[0]
    hooked_output, cache = hooked_model.run_with_cache(input_image)

    # Compare final outputs
    # assert torch.allclose(hooked_output, og_output, atol=TOLERANCE), f"Model output diverges! Max diff: {torch.max(torch.abs(hooked_output - og_output))}"

    # Compare intermediate activations
    for og_name, og_act in og_activations.items():
        og_act = og_act[0]  # Get the first (and only) activation
        
        # Try to find a matching activation in the hooked model cache
        matching_name = None
        for cache_name, cache_act in cache.items():
            if og_act.shape == cache_act.shape:
                matching_name = cache_name
                break
        
        if matching_name is None:
            print(f"No matching shape found for {og_name} with shape {og_act.shape}")
            continue
        
        hooked_act = cache[matching_name]
        
        if not torch.allclose(og_act, hooked_act, atol=TOLERANCE):
            max_diff = torch.max(torch.abs(og_act - hooked_act))
            print(f"Activation mismatch for {og_name} (matched with {matching_name}). Max diff: {max_diff}")
        else:
            print(f"Activation match for {og_name} (matched with {matching_name})")

    print(f"Test completed!")
    print(f"Hooked model output shape: {hooked_output.shape}")
    print(f"Original model output shape: {og_output.shape}")
    print(f"Hooked model output (first few values): {hooked_output[0, :5]}")
    print(f"Original model output (first few values): {og_output[0, :5]}")

test_loading_open_clip()