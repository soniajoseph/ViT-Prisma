

# load model
import torch
from vit_prisma.models.base_vit import HookedViT

from transformers import ViTConfig

from transformers import PretrainedConfig
import torch.nn as nn

from functools import partial

import math


from jepa.models.vision_transformer import VisionTransformer

from jepa.models.utils.patch_embed import PatchEmbed, PatchEmbed3D
from jepa.models.utils.modules import Block
from jepa.models.utils.pos_embs import get_2d_sincos_pos_embed, get_3d_sincos_pos_embed
from jepa.utils.tensors import trunc_normal_
from jepa.masks.utils import apply_masks
from jepa.models.utils.multimask import MultiMaskWrapper, PredictorMultiMaskWrapper




def generate_random_video(batch_size=1, channels=3, num_frames=16, height=224, width=224):   # W
    # Generate random tensor
    random_video = torch.randn(batch_size, channels, num_frames, height, width)
    random_video = random_video.to(DEVICE)
    return random_video



def get_all_layer_outputs(model, input_tensor):
    layer_outputs = []
    layer_names = []

    def hook_fn(module, input, output):
        layer_outputs.append(output)
        layer_names.append(type(module).__name__)

    hooks = []
    for name, module in model.named_modules():
        hooks.append(module.register_forward_hook(hook_fn))

    with torch.no_grad():
        model(input_tensor)

    for hook in hooks:
        hook.remove()

    return layer_outputs, layer_names

def inspect_model_structures(encoder, hooked_encoder, random_video):
    print("\n=== Original Model First Block Structure ===")
    # Get first block of original model
    first_block = encoder.blocks[0]
    for name, module in first_block.named_modules():
        print(f"Original first block layer: {name}")
    
    print("\n=== Hooked Model First Block Structure ===")
    # Get first block of hooked model
    first_block_hooked = hooked_encoder.blocks[0]
    for name, module in first_block_hooked.named_modules():
        print(f"Hooked first block layer: {name}")
    
    print("\n=== Model Component Comparison ===")
    print(f"Original embedding dim: {encoder.embed_dim}")
    print(f"Hooked embedding dim: {hooked_encoder.cfg.d_model}")
    print(f"Original num heads: {encoder.num_heads}")
    print(f"Hooked num heads: {hooked_encoder.cfg.n_heads}")
    print(f"Original num layers: {len(encoder.blocks)}")
    print(f"Hooked num layers: {len(hooked_encoder.blocks)}")

def compare_models(encoder, hooked_encoder, random_video, tolerance=1e-4):
    print(f"\nComparing model activations with tolerance {tolerance}...")
    
    # Dictionary to store original model activations
    original_activations = {}
    
    # Hook functions to capture activations in original model
    def get_activation_hook(name):
        def hook(module, input, output):
            original_activations[name] = output
        return hook
    
    # Register hooks in original model
    hooks = []
    for i, block in enumerate(encoder.blocks):
        # Attention hooks
        hooks.append(block.attn.proj.register_forward_hook(
            get_activation_hook(f'blocks.{i}.attn_out')))
        # MLP hooks
        hooks.append(block.mlp.fc2.register_forward_hook(
            get_activation_hook(f'blocks.{i}.mlp_out')))
        # Block output hooks
        hooks.append(block.register_forward_hook(
            get_activation_hook(f'blocks.{i}.block_out')))
    
    # Get hooked model outputs and cache
    final_output_hooked, cache = hooked_encoder.run_with_cache(random_video)
    
    # Get original model outputs (this will populate original_activations)
    with torch.no_grad():
        final_output_og = encoder(random_video)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    print("\nComparing intermediate activations:")
    
    # Compare transformer blocks
    for i in range(len(encoder.blocks)):
        print(f"\nBlock {i}:")
        
        # Compare attention outputs
        if f'blocks.{i}.hook_attn_out' in cache:
            attn_out_hooked = cache[f'blocks.{i}.hook_attn_out']
            attn_out_orig = original_activations[f'blocks.{i}.attn_out']
            max_diff = torch.max(torch.abs(attn_out_hooked - attn_out_orig))
            print(f"Attention output max difference: {max_diff:.2e}")
            if max_diff > tolerance:
                print(f"WARNING: Attention difference ({max_diff:.2e}) exceeds tolerance ({tolerance:.2e})")
        
        # Compare MLP outputs
        if f'blocks.{i}.hook_mlp_out' in cache:
            mlp_out_hooked = cache[f'blocks.{i}.hook_mlp_out']
            mlp_out_orig = original_activations[f'blocks.{i}.mlp_out']
            max_diff = torch.max(torch.abs(mlp_out_hooked - mlp_out_orig))
            print(f"MLP output max difference: {max_diff:.2e}")
            if max_diff > tolerance:
                print(f"WARNING: MLP difference ({max_diff:.2e}) exceeds tolerance ({tolerance:.2e})")
        
        # Compare block outputs
        if f'blocks.{i}.hook_resid_post' in cache:
            block_out_hooked = cache[f'blocks.{i}.hook_resid_post']
            block_out_orig = original_activations[f'blocks.{i}.block_out']
            max_diff = torch.max(torch.abs(block_out_hooked - block_out_orig))
            print(f"Block output max difference: {max_diff:.2e}")
            if max_diff > tolerance:
                print(f"WARNING: Block difference ({max_diff:.2e}) exceeds tolerance ({tolerance:.2e})")

    # print shapes of output
    print(f"Final output shape: {final_output_og.shape}")
    print(f"Hooked final output shape: {final_output_hooked.shape}")

    # Check final outputs
    max_diff = torch.max(torch.abs(final_output_og - final_output_hooked))
    print(f"\nFinal output max difference: {max_diff:.2e}")
    if max_diff > tolerance:
        print(f"WARNING: Final output difference ({max_diff:.2e}) exceeds tolerance ({tolerance:.2e})")

    return max_diff <= tolerance

# Use in your main code

# # Check ImageFolder's automatic mapping
# from torchvision.datasets import ImageFolder

# val_dir = 

# imagefolder_dataset = ImageFolder(val_dir)
# imagefolder_classes = imagefolder_dataset.classes
# print("ImageFolder first 5 class indices:")
# for i in range(5):
#     print(f"{i}: {imagefolder_classes[i]}")


DEVICE = 'cuda'
path = '/network/scratch/s/sonia.joseph/jepa_models/github_models/vit-l-16/vitl16.pth.tar'

# Load original model
model = torch.load(path)
encoder = VisionTransformer(
    patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,
    qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), tubelet_size=2, num_frames=16, use_sdpa=True,
)
encoder.eval()
encoder.to(DEVICE)
encoder_dict = model['encoder']
new_state_dict = {k.replace('module.', ''): v for k, v in encoder_dict.items()}
new_state_dict = {k.replace('backbone.', ''): v for k, v in new_state_dict.items()}
encoder.load_state_dict(new_state_dict)

# Load hooked model
model_name = "vjepa_v1_vit_huge_patch16_224"
hooked_encoder = HookedViT.from_pretrained(
    model_name, is_timm=False, is_clip=False, fold_ln=False, center_writing_weights=False
)
hooked_encoder.to(DEVICE)
hooked_encoder.eval()


import einops
import torch

# Extract and reshape original QKV weights
W_qkv_original = encoder.blocks[0].attn.qkv.weight  # Shape: (3 * n_heads * d_head, d_model)

# Reshape and split into Q, K, V
W_qkv_original_reshaped = einops.rearrange(
    W_qkv_original, 
    "(three h dh) d -> three h d dh", 
    three=3, 
    h=16,
    dh=hooked_encoder.cfg.d_head,
)

W_Q_original, W_K_original, W_V_original = torch.unbind(W_qkv_original_reshaped, dim=0)

# Get the hooked model's weights (already correctly reshaped)
W_Q_hooked = hooked_encoder.blocks[0].attn.W_Q
W_K_hooked = hooked_encoder.blocks[0].attn.W_K
W_V_hooked = hooked_encoder.blocks[0].attn.W_V

# Check shape consistency
print(f"Q original shape: {W_Q_original.shape}, Hooked shape: {W_Q_hooked.shape}")
print(f"K original shape: {W_K_original.shape}, Hooked shape: {W_K_hooked.shape}")
print(f"V original shape: {W_V_original.shape}, Hooked shape: {W_V_hooked.shape}")

# Compute max differences
print(f"Q weight difference: {torch.max(torch.abs(W_Q_original - W_Q_hooked)).item()}")
print(f"K weight difference: {torch.max(torch.abs(W_K_original - W_K_hooked)).item()}")
print(f"V weight difference: {torch.max(torch.abs(W_V_original - W_V_hooked)).item()}")



# check mlp weights of first block
print(new_state_dict.keys())
w1 = encoder.blocks[0].mlp.fc1.weight
w2 = hooked_encoder.blocks[0].mlp.W_in
# print shapes
print(f"Original first block MLP weight shape: {w1.shape}")
print(f"Hooked first block MLP weight shape: {w2.shape}")
print(f"Original first block MLP weight difference: {torch.max(torch.abs(w1 - w2.T)).item()}")

# for name, param in encoder.named_parameters():
#     print(f"{name}: {param.dtype}")


# for name, param in hooked_encoder.named_parameters():
#     print(f"{name}: {param.dtype}")



# checkpoint_keys = set(new_state_dict.keys())

# prisma_keys = set(hooked_encoder.state_dict().keys())

# missing_keys = prisma_keys - checkpoint_keys
# extra_keys = checkpoint_keys - prisma_keys

# print(f"Missing Keys: {missing_keys}")
# print(f"Extra Keys: {extra_keys}")


# Check difference
random_video = generate_random_video()

inspect_model_structures(encoder, hooked_encoder, random_video)

# Generate random input and compare models
compare_models(encoder, hooked_encoder, random_video)

# model = torch.load(path)
# encoder = VisionTransformer(
#     patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,
#     qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), tubelet_size=2, num_frames=16
# )
# encoder = MultiMaskWrapper(encoder)
# encoder.to(DEVICE)

# encoder_dict = model['encoder']
# new_state_dict = {}
# for key, value in encoder_dict.items():
#     if key.startswith('module.'):
#         new_key = key[7:]  # Remove 'module.' prefix
#         new_state_dict[new_key] = value
#     else:
#         new_state_dict[key] = value

# # Load the modified state dict
# encoder.load_state_dict(new_state_dict)

# model_name = 'vjepa'
# hooked_encoder = HookedViT.from_pretrained(
#     model_name, is_timm=False, is_clip=False, fold_ln=False, center_writing_weights=False
# )
# hooked_encoder.to(DEVICE)
# hooked_encoder.eval()

# random_video = generate_random_video()


# og_output = encoder(random_video)
# hooked_output = hooked_encoder(random_video)

# # check tolerance
# # also print difference
# assert torch.allclose(og_output, hooked_output, atol=1e-4), f"Outputs do not match within tolerance. Max difference: {torch.max(torch.abs(og_output - hooked_output)).item()}"


