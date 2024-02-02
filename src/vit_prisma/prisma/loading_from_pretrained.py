"""
Reference:
https://github.com/neelnanda-io/TransformerLens/blob/main/transformer_lens/loading_from_pretrained.py

Preserves most of the original functionality with necessary modifications for ViTs and pretrained ViT models.
"""

import logging

from transformers import AutoConfig
from transformers import ViTForImageClassification

import timm
from vit_prisma.configs.HookedViTConfig import HookedViTConfig

import torch

from typing import Dict

import einops

def convert_timm_weigthts(
        old_state_dict,
        cfg: HookedViTConfig,
):
    
    # OLD odict_keys(['cls_token', 'pos_embed', 'patch_embed.proj.weight', 'patch_embed.proj.bias', 'blocks.0.norm1.weight', 'blocks.0.norm1.bias', 'blocks.0.attn.qkv.weight', 'blocks.0.attn.qkv.bias', 'blocks.0.attn.proj.weight', 'blocks.0.attn.proj.bias', 'blocks.0.norm2.weight', 'blocks.0.norm2.bias', 'blocks.0.mlp.fc1.weight', 'blocks.0.mlp.fc1.bias', 'blocks.0.mlp.fc2.weight', 'blocks.0.mlp.fc2.bias', 'blocks.1.norm1.weight', 'blocks.1.norm1.bias', 'blocks.1.attn.qkv.weight', 'blocks.1.attn.qkv.bias', 'blocks.1.attn.proj.weight', 'blocks.1.attn.proj.bias', 'blocks.1.norm2.weight', 'blocks.1.norm2.bias', 'blocks.1.mlp.fc1.weight', 'blocks.1.mlp.fc1.bias', 'blocks.1.mlp.fc2.weight', 'blocks.1.mlp.fc2.bias', 'blocks.2.norm1.weight', 'blocks.2.norm1.bias', 'blocks.2.attn.qkv.weight', 'blocks.2.attn.qkv.bias', 'blocks.2.attn.proj.weight', 'blocks.2.attn.proj.bias', 'blocks.2.norm2.weight', 'blocks.2.norm2.bias', 'blocks.2.mlp.fc1.weight', 'blocks.2.mlp.fc1.bias', 'blocks.2.mlp.fc2.weight', 'blocks.2.mlp.fc2.bias', 'blocks.3.norm1.weight', 'blocks.3.norm1.bias', 'blocks.3.attn.qkv.weight', 'blocks.3.attn.qkv.bias', 'blocks.3.attn.proj.weight', 'blocks.3.attn.proj.bias', 'blocks.3.norm2.weight', 'blocks.3.norm2.bias', 'blocks.3.mlp.fc1.weight', 'blocks.3.mlp.fc1.bias', 'blocks.3.mlp.fc2.weight', 'blocks.3.mlp.fc2.bias', 'blocks.4.norm1.weight', 'blocks.4.norm1.bias', 'blocks.4.attn.qkv.weight', 'blocks.4.attn.qkv.bias', 'blocks.4.attn.proj.weight', 'blocks.4.attn.proj.bias', 'blocks.4.norm2.weight', 'blocks.4.norm2.bias', 'blocks.4.mlp.fc1.weight', 'blocks.4.mlp.fc1.bias', 'blocks.4.mlp.fc2.weight', 'blocks.4.mlp.fc2.bias', 'blocks.5.norm1.weight', 'blocks.5.norm1.bias', 'blocks.5.attn.qkv.weight', 'blocks.5.attn.qkv.bias', 'blocks.5.attn.proj.weight', 'blocks.5.attn.proj.bias', 'blocks.5.norm2.weight', 'blocks.5.norm2.bias', 'blocks.5.mlp.fc1.weight', 'blocks.5.mlp.fc1.bias', 'blocks.5.mlp.fc2.weight', 'blocks.5.mlp.fc2.bias', 'blocks.6.norm1.weight', 'blocks.6.norm1.bias', 'blocks.6.attn.qkv.weight', 'blocks.6.attn.qkv.bias', 'blocks.6.attn.proj.weight', 'blocks.6.attn.proj.bias', 'blocks.6.norm2.weight', 'blocks.6.norm2.bias', 'blocks.6.mlp.fc1.weight', 'blocks.6.mlp.fc1.bias', 'blocks.6.mlp.fc2.weight', 'blocks.6.mlp.fc2.bias', 'blocks.7.norm1.weight', 'blocks.7.norm1.bias', 'blocks.7.attn.qkv.weight', 'blocks.7.attn.qkv.bias', 'blocks.7.attn.proj.weight', 'blocks.7.attn.proj.bias', 'blocks.7.norm2.weight', 'blocks.7.norm2.bias', 'blocks.7.mlp.fc1.weight', 'blocks.7.mlp.fc1.bias', 'blocks.7.mlp.fc2.weight', 'blocks.7.mlp.fc2.bias', 'blocks.8.norm1.weight', 'blocks.8.norm1.bias', 'blocks.8.attn.qkv.weight', 'blocks.8.attn.qkv.bias', 'blocks.8.attn.proj.weight', 'blocks.8.attn.proj.bias', 'blocks.8.norm2.weight', 'blocks.8.norm2.bias', 'blocks.8.mlp.fc1.weight', 'blocks.8.mlp.fc1.bias', 'blocks.8.mlp.fc2.weight', 'blocks.8.mlp.fc2.bias', 'blocks.9.norm1.weight', 'blocks.9.norm1.bias', 'blocks.9.attn.qkv.weight', 'blocks.9.attn.qkv.bias', 'blocks.9.attn.proj.weight', 'blocks.9.attn.proj.bias', 'blocks.9.norm2.weight', 'blocks.9.norm2.bias', 'blocks.9.mlp.fc1.weight', 'blocks.9.mlp.fc1.bias', 'blocks.9.mlp.fc2.weight', 'blocks.9.mlp.fc2.bias', 'blocks.10.norm1.weight', 'blocks.10.norm1.bias', 'blocks.10.attn.qkv.weight', 'blocks.10.attn.qkv.bias', 'blocks.10.attn.proj.weight', 'blocks.10.attn.proj.bias', 'blocks.10.norm2.weight', 'blocks.10.norm2.bias', 'blocks.10.mlp.fc1.weight', 'blocks.10.mlp.fc1.bias', 'blocks.10.mlp.fc2.weight', 'blocks.10.mlp.fc2.bias', 'blocks.11.norm1.weight', 'blocks.11.norm1.bias', 'blocks.11.attn.qkv.weight', 'blocks.11.attn.qkv.bias', 'blocks.11.attn.proj.weight', 'blocks.11.attn.proj.bias', 'blocks.11.norm2.weight', 'blocks.11.norm2.bias', 'blocks.11.mlp.fc1.weight', 'blocks.11.mlp.fc1.bias', 'blocks.11.mlp.fc2.weight', 'blocks.11.mlp.fc2.bias', 'norm.weight', 'norm.bias', 'head.weight', 'head.bias'])
    # NEW odict_keys(['cls_token', 'embed.proj.weight', 'pos_embed.W_pos', 'blocks.0.ln1.w', 'blocks.0.ln1.b', 'blocks.0.ln2.w', 'blocks.0.ln2.b', 'blocks.0.attn.W_Q', 'blocks.0.attn.W_K', 'blocks.0.attn.W_V', 'blocks.0.attn.W_O', 'blocks.0.attn.b_Q', 'blocks.0.attn.b_K', 'blocks.0.attn.b_V', 'blocks.0.attn.b_O', 'blocks.0.mlp.W_in', 'blocks.0.mlp.b_in', 'blocks.0.mlp.W_out', 'blocks.0.mlp.b_out', 'blocks.1.ln1.w', 'blocks.1.ln1.b', 'blocks.1.ln2.w', 'blocks.1.ln2.b', 'blocks.1.attn.W_Q', 'blocks.1.attn.W_K', 'blocks.1.attn.W_V', 'blocks.1.attn.W_O', 'blocks.1.attn.b_Q', 'blocks.1.attn.b_K', 'blocks.1.attn.b_V', 'blocks.1.attn.b_O', 'blocks.1.mlp.W_in', 'blocks.1.mlp.b_in', 'blocks.1.mlp.W_out', 'blocks.1.mlp.b_out', 'blocks.2.ln1.w', 'blocks.2.ln1.b', 'blocks.2.ln2.w', 'blocks.2.ln2.b', 'blocks.2.attn.W_Q', 'blocks.2.attn.W_K', 'blocks.2.attn.W_V', 'blocks.2.attn.W_O', 'blocks.2.attn.b_Q', 'blocks.2.attn.b_K', 'blocks.2.attn.b_V', 'blocks.2.attn.b_O', 'blocks.2.mlp.W_in', 'blocks.2.mlp.b_in', 'blocks.2.mlp.W_out', 'blocks.2.mlp.b_out', 'blocks.3.ln1.w', 'blocks.3.ln1.b', 'blocks.3.ln2.w', 'blocks.3.ln2.b', 'blocks.3.attn.W_Q', 'blocks.3.attn.W_K', 'blocks.3.attn.W_V', 'blocks.3.attn.W_O', 'blocks.3.attn.b_Q', 'blocks.3.attn.b_K', 'blocks.3.attn.b_V', 'blocks.3.attn.b_O', 'blocks.3.mlp.W_in', 'blocks.3.mlp.b_in', 'blocks.3.mlp.W_out', 'blocks.3.mlp.b_out', 'blocks.4.ln1.w', 'blocks.4.ln1.b', 'blocks.4.ln2.w', 'blocks.4.ln2.b', 'blocks.4.attn.W_Q', 'blocks.4.attn.W_K', 'blocks.4.attn.W_V', 'blocks.4.attn.W_O', 'blocks.4.attn.b_Q', 'blocks.4.attn.b_K', 'blocks.4.attn.b_V', 'blocks.4.attn.b_O', 'blocks.4.mlp.W_in', 'blocks.4.mlp.b_in', 'blocks.4.mlp.W_out', 'blocks.4.mlp.b_out', 'blocks.5.ln1.w', 'blocks.5.ln1.b', 'blocks.5.ln2.w', 'blocks.5.ln2.b', 'blocks.5.attn.W_Q', 'blocks.5.attn.W_K', 'blocks.5.attn.W_V', 'blocks.5.attn.W_O', 'blocks.5.attn.b_Q', 'blocks.5.attn.b_K', 'blocks.5.attn.b_V', 'blocks.5.attn.b_O', 'blocks.5.mlp.W_in', 'blocks.5.mlp.b_in', 'blocks.5.mlp.W_out', 'blocks.5.mlp.b_out', 'blocks.6.ln1.w', 'blocks.6.ln1.b', 'blocks.6.ln2.w', 'blocks.6.ln2.b', 'blocks.6.attn.W_Q', 'blocks.6.attn.W_K', 'blocks.6.attn.W_V', 'blocks.6.attn.W_O', 'blocks.6.attn.b_Q', 'blocks.6.attn.b_K', 'blocks.6.attn.b_V', 'blocks.6.attn.b_O', 'blocks.6.mlp.W_in', 'blocks.6.mlp.b_in', 'blocks.6.mlp.W_out', 'blocks.6.mlp.b_out', 'blocks.7.ln1.w', 'blocks.7.ln1.b', 'blocks.7.ln2.w', 'blocks.7.ln2.b', 'blocks.7.attn.W_Q', 'blocks.7.attn.W_K', 'blocks.7.attn.W_V', 'blocks.7.attn.W_O', 'blocks.7.attn.b_Q', 'blocks.7.attn.b_K', 'blocks.7.attn.b_V', 'blocks.7.attn.b_O', 'blocks.7.mlp.W_in', 'blocks.7.mlp.b_in', 'blocks.7.mlp.W_out', 'blocks.7.mlp.b_out', 'blocks.8.ln1.w', 'blocks.8.ln1.b', 'blocks.8.ln2.w', 'blocks.8.ln2.b', 'blocks.8.attn.W_Q', 'blocks.8.attn.W_K', 'blocks.8.attn.W_V', 'blocks.8.attn.W_O', 'blocks.8.attn.b_Q', 'blocks.8.attn.b_K', 'blocks.8.attn.b_V', 'blocks.8.attn.b_O', 'blocks.8.mlp.W_in', 'blocks.8.mlp.b_in', 'blocks.8.mlp.W_out', 'blocks.8.mlp.b_out', 'blocks.9.ln1.w', 'blocks.9.ln1.b', 'blocks.9.ln2.w', 'blocks.9.ln2.b', 'blocks.9.attn.W_Q', 'blocks.9.attn.W_K', 'blocks.9.attn.W_V', 'blocks.9.attn.W_O', 'blocks.9.attn.b_Q', 'blocks.9.attn.b_K', 'blocks.9.attn.b_V', 'blocks.9.attn.b_O', 'blocks.9.mlp.W_in', 'blocks.9.mlp.b_in', 'blocks.9.mlp.W_out', 'blocks.9.mlp.b_out', 'blocks.10.ln1.w', 'blocks.10.ln1.b', 'blocks.10.ln2.w', 'blocks.10.ln2.b', 'blocks.10.attn.W_Q', 'blocks.10.attn.W_K', 'blocks.10.attn.W_V', 'blocks.10.attn.W_O', 'blocks.10.attn.b_Q', 'blocks.10.attn.b_K', 'blocks.10.attn.b_V', 'blocks.10.attn.b_O', 'blocks.10.mlp.W_in', 'blocks.10.mlp.b_in', 'blocks.10.mlp.W_out', 'blocks.10.mlp.b_out', 'blocks.11.ln1.w', 'blocks.11.ln1.b', 'blocks.11.ln2.w', 'blocks.11.ln2.b', 'blocks.11.attn.W_Q', 'blocks.11.attn.W_K', 'blocks.11.attn.W_V', 'blocks.11.attn.W_O', 'blocks.11.attn.b_Q', 'blocks.11.attn.b_K', 'blocks.11.attn.b_V', 'blocks.11.attn.b_O', 'blocks.11.mlp.W_in', 'blocks.11.mlp.b_in', 'blocks.11.mlp.W_out', 'blocks.11.mlp.b_out', 'ln_final.w', 'ln_final.b', 'head.weight', 'head.bias'])

    new_state_dict = {}
    new_state_dict["cls_token"] = old_state_dict["cls_token"]
    new_state_dict["pos_embed.W_pos"] = old_state_dict["pos_embed"]
    new_state_dict["embed.proj.weight"] = old_state_dict["patch_embed.proj.weight"]
    # new_state_dict["embed.proj.bias"] = old_state_dict["patch_embed.proj.bias"] 
    new_state_dict["ln_final.w"] = old_state_dict["norm.weight"]
    new_state_dict["ln_final.b"] = old_state_dict["norm.bias"]

    for layer in range(cfg.n_layers):
        layer_key = f"blocks.{layer}" 
        new_state_dict[f"{layer_key}.ln1.w"] = old_state_dict[f"{layer_key}.norm1.weight"]
        new_state_dict[f"{layer_key}.ln1.b"] = old_state_dict[f"{layer_key}.norm1.bias"]
        new_state_dict[f"{layer_key}.ln2.w"] = old_state_dict[f"{layer_key}.norm2.weight"]
        new_state_dict[f"{layer_key}.ln2.b"] = old_state_dict[f"{layer_key}.norm2.bias"]

        W = old_state_dict[f"{layer_key}.attn.qkv.weight"]
        W_Q, W_K, W_V = torch.tensor_split(W, 3, dim=0)
        W_Q = einops.rearrange(W_Q, "(i h) m->i m h", h=cfg.n_heads)
        W_K = einops.rearrange(W_K, "(i h) m->i m h", h=cfg.n_heads)
        W_V = einops.rearrange(W_V, "(i h) m->i m h", h=cfg.n_heads)
        new_state_dict[f"{layer_key}.attn.W_Q"] = W_Q
        new_state_dict[f"{layer_key}.attn.W_K"] = W_K
        new_state_dict[f"{layer_key}.attn.W_V"] = W_V

        W_O = old_state_dict[f"{layer_key}.attn.proj.weight"]
        W_O = einops.rearrange(W_O, "m (i h)->i h m", i=cfg.n_heads)
        new_state_dict[f"{layer_key}.attn.W_O"] = W_O

        attn_bias = old_state_dict[f"{layer_key}.attn.qkv.bias"]
        b_Q, b_K, b_V = torch.tensor_split(attn_bias, 3, dim=0)
        b_Q = einops.rearrange(b_Q, "(i h) -> i h", h=cfg.n_heads)
        b_K = einops.rearrange(b_K, "(i h) -> i h", h=cfg.n_heads)
        b_V = einops.rearrange(b_V, "(i h) -> i h", h=cfg.n_heads)
        new_state_dict[f"{layer_key}.attn.b_Q"] = b_Q
        new_state_dict[f"{layer_key}.attn.b_K"] = b_K
        new_state_dict[f"{layer_key}.attn.b_V"] = b_V

        b_O = old_state_dict[f"{layer_key}.attn.proj.bias"]
        b_O = einops.rearrange(b_O, "m -> m")
        new_state_dict[f"{layer_key}.attn.b_O"] = b_O

        new_state_dict[f"{layer_key}.mlp.W_in"] = old_state_dict[f"{layer_key}.mlp.fc1.weight"]
        new_state_dict[f"{layer_key}.mlp.b_in"] = old_state_dict[f"{layer_key}.mlp.fc1.bias"]
        new_state_dict[f"{layer_key}.mlp.W_out"] = old_state_dict[f"{layer_key}.mlp.fc2.weight"]
        new_state_dict[f"{layer_key}.mlp.b_out"] = old_state_dict[f"{layer_key}.mlp.fc2.bias"]

    new_state_dict["head.weight"] = old_state_dict["head.weight"]
    new_state_dict["head.bias"] = old_state_dict["head.bias"]

    return new_state_dict

def get_pretrained_state_dict(
    official_model_name: str,
    is_timm: bool,
    cfg: HookedViTConfig,
    hf_model=None,
    dtype: torch.dtype = torch.float32,
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """
    Loads in the model weights for a pretrained model, and processes them to
    have the HookedTransformer parameter names and shapes. Supports checkpointed
    models (and expects the checkpoint info to be stored in the config object)

    hf_model: Optionally, a HuggingFace model object. If provided, we will use
        these weights rather than reloading the model.
    dtype: The dtype to load the HuggingFace model in.
    kwargs: Other optional arguments passed to HuggingFace's from_pretrained.
        Also given to other HuggingFace functions when compatible.
    """
    if "torch_dtype" in kwargs:
        dtype = kwargs["torch_dtype"]
        del kwargs["torch_dtype"]
    # official_model_name = get_official_model_name(official_model_name)
    # if official_model_name.startswith(NEED_REMOTE_CODE_MODELS) and not kwargs.get(
    #     "trust_remote_code", False
    # ):
    #     logging.warning(
    #         f"Loading model {official_model_name} state dict requires setting trust_remote_code=True"
    #     )
    #     kwargs["trust_remote_code"] = True
        
    try:
        if is_timm:
            hf_model = timm.create_model(official_model_name, pretrained=True)
        else:
            hf_model = ViTForImageClassification.from_pretrained(
                    official_model_name, torch_dtype=dtype, **kwargs
            )

            # Load model weights, and fold in layer norm weights

        for param in hf_model.parameters():
            param.requires_grad = False

        # state_dict = None # Conversion of state dict to HookedTransformer format       
        state_dict = convert_timm_weigthts(hf_model.state_dict(), cfg)
                
        return state_dict

    except:
        raise ValueError(
            f"Loading weights from the architecture is not currently supported: {cfg.original_architecture}, generated from model name {cfg.model_name}. Feel free to open an issue on GitHub to request this feature."
        )

def fill_missing_keys(model, state_dict):
    """Takes in a state dict from a pretrained model, and fills in any missing keys with the default initialization.

    This function is assumed to be run before weights are initialized.

    Args:
        state_dict (dict): State dict from a pretrained model

    Returns:
        dict: State dict with missing keys filled in
    """
    # Get the default state dict
    default_state_dict = model.state_dict()
    # Get the keys that are missing from the pretrained model
    missing_keys = set(default_state_dict.keys()) - set(state_dict.keys())
    # Fill in the missing keys with the default initialization
    for key in missing_keys:
        if "hf_model" in key:
            # Skip keys that are from the HuggingFace model, if loading from HF.
            continue
        if "W_" in key:
            logging.warning(
                "Missing key for a weight matrix in pretrained, filled in with an empty tensor: {}".format(
                    key
                )
            )
        state_dict[key] = default_state_dict[key]
    return state_dict

def convert_pretrained_model_config(model: str, is_timm: bool = True) -> HookedViTConfig:

    if is_timm:
        model = timm.create_model(model)
        hf_config = AutoConfig.from_pretrained(model.default_cfg['hf_hub_id'])
    else:
        hf_config = AutoConfig.from_pretrained(model)

    pretrained_config = {
                    'n_layers' : hf_config.num_hidden_layers,
                    'd_model' : hf_config.hidden_size,
                    'd_head' : hf_config.hidden_size // hf_config.num_attention_heads,
                    'model_name' : hf_config._name_or_path,
                    'n_heads' : hf_config.num_attention_heads,
                    'd_mlp' : hf_config.intermediate_size,
                    'activation_name' : hf_config.hidden_act,
                    'eps' : hf_config.layer_norm_eps,
                    'original_architecture' : hf_config.architecture,
                    'initializer_range' : hf_config.initializer_range,
                    'n_channels' : hf_config.num_channels,
                    'patch_size' : hf_config.patch_size,
                    'image_size' : hf_config.image_size,
                    'n_classes' : hf_config.num_classes,
                    'n_params' : sum(p.numel() for p in model.parameters() if p.requires_grad) if is_timm else None,
                }

    return HookedViTConfig.from_dict(pretrained_config)
