import pytest
import torch

from vit_prisma.models.base_vit import HookedViT
from vit_prisma.configs.HookedViTConfig import HookedViTConfig

#Test taken from transformerlens with minor modifications

batch_size = 2
channels = 3
height = 224
width = 224
input_images = torch.rand((batch_size, channels, height, width))

# instantiate a untrained network
n_layers = 1
d_head = 8
d_model = 8
d_mlp=8

model = HookedViT(HookedViTConfig(n_layers,d_head,d_model, d_mlp, return_type="logits", activation_name="solu_ln"))

act_names_in_cache = [
    "hook_embed",
    "hook_pos_embed",
    "blocks.0.hook_resid_pre",
    "blocks.0.ln1.hook_scale",
    "blocks.0.ln1.hook_normalized",
    "blocks.0.attn.hook_q",
    "blocks.0.attn.hook_k",
    "blocks.0.attn.hook_v",
    "blocks.0.attn.hook_attn_scores",
    "blocks.0.attn.hook_pattern",
    "blocks.0.attn.hook_z",
    "blocks.0.hook_attn_out",
    "blocks.0.hook_resid_mid",
    "blocks.0.ln2.hook_scale",
    "blocks.0.ln2.hook_normalized",
    "blocks.0.mlp.hook_pre",
    "blocks.0.mlp.hook_mid",
    "blocks.0.mlp.ln.hook_scale",
    "blocks.0.mlp.ln.hook_normalized",
    "blocks.0.mlp.hook_post",
    "blocks.0.hook_mlp_out",
    "blocks.0.hook_resid_post",
    "ln_final.hook_scale",
    "ln_final.hook_normalized",
]


def test_cache_hook_names():
    _, cache = model.run_with_cache(input_images)
    assert list(cache.keys()) == act_names_in_cache