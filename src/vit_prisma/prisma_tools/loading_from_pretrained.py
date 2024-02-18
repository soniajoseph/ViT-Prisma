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

"""
Official model names from Huggingface.
"""

def convert_timm_weights(
        old_state_dict,
        cfg: HookedViTConfig,
):

    new_state_dict = {}
    new_state_dict["cls_token"] = old_state_dict["cls_token"]
    new_state_dict["pos_embed.W_pos"] = old_state_dict["pos_embed"].squeeze(0)
    new_state_dict["embed.proj.weight"] = old_state_dict["patch_embed.proj.weight"]
    new_state_dict["embed.proj.bias"] = old_state_dict["patch_embed.proj.bias"] 
    new_state_dict["ln_final.w"] = old_state_dict["norm.weight"]
    new_state_dict["ln_final.b"] = old_state_dict["norm.bias"]

    for layer in range(cfg.n_layers):
        layer_key = f"blocks.{layer}" 
        new_state_dict[f"{layer_key}.ln1.w"] = old_state_dict[f"{layer_key}.norm1.weight"]
        new_state_dict[f"{layer_key}.ln1.b"] = old_state_dict[f"{layer_key}.norm1.bias"]
        new_state_dict[f"{layer_key}.ln2.w"] = old_state_dict[f"{layer_key}.norm2.weight"]
        new_state_dict[f"{layer_key}.ln2.b"] = old_state_dict[f"{layer_key}.norm2.bias"]

        W = old_state_dict[f"{layer_key}.attn.qkv.weight"]
        W_reshape = einops.rearrange( W, "(three h dh) d ->three h d dh" , three=3, h=cfg.n_heads, d=cfg.d_model, dh=cfg.d_head)
        W_Q, W_K, W_V = torch.unbind(W_reshape, dim=0)
        new_state_dict[f"{layer_key}.attn.W_Q"] = W_Q
        new_state_dict[f"{layer_key}.attn.W_K"] = W_K
        new_state_dict[f"{layer_key}.attn.W_V"] = W_V

        W_O = old_state_dict[f"{layer_key}.attn.proj.weight"]
        W_O = einops.rearrange(W_O, "m (i h)->i h m", i=cfg.n_heads)
        new_state_dict[f"{layer_key}.attn.W_O"] = W_O

        attn_bias = old_state_dict[f"{layer_key}.attn.qkv.bias"]
        attn_bias_reshape = einops.rearrange(attn_bias, "(three h dh) -> three h dh", three=3, h=cfg.n_heads, dh=cfg.d_head)
        b_Q, b_K, b_V = torch.unbind(attn_bias_reshape, dim=0)
        new_state_dict[f"{layer_key}.attn.b_Q"] = b_Q
        new_state_dict[f"{layer_key}.attn.b_K"] = b_K
        new_state_dict[f"{layer_key}.attn.b_V"] = b_V

        b_O = old_state_dict[f"{layer_key}.attn.proj.bias"]
        new_state_dict[f"{layer_key}.attn.b_O"] = b_O

        new_state_dict[f"{layer_key}.mlp.b_in"] = old_state_dict[f"{layer_key}.mlp.fc1.bias"]
        new_state_dict[f"{layer_key}.mlp.b_out"] = old_state_dict[f"{layer_key}.mlp.fc2.bias"]

        mlp_W_in = old_state_dict[f"{layer_key}.mlp.fc1.weight"]
        mlp_W_in = einops.rearrange(mlp_W_in, "m d -> d m")
        new_state_dict[f"{layer_key}.mlp.W_in"] = mlp_W_in

        mlp_W_out = old_state_dict[f"{layer_key}.mlp.fc2.weight"]
        mlp_W_out = einops.rearrange(mlp_W_out, "d m -> m d")
        new_state_dict[f"{layer_key}.mlp.W_out"] = mlp_W_out



    new_state_dict["head.W_H"] = einops.rearrange(old_state_dict["head.weight"], "c d -> d c")
    new_state_dict["head.b_H"] = old_state_dict["head.bias"]

    

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
        state_dict = convert_timm_weights(hf_model.state_dict(), cfg)
                
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

def convert_pretrained_model_config(model_name: str, is_timm: bool = True, is_clip: bool = False) -> HookedViTConfig:
    
    

    if is_timm:
        model = timm.create_model(model_name)
        hf_config = AutoConfig.from_pretrained(model.default_cfg['hf_hub_id'])
    elif is_clip: # Extract vision encoder from dual-encoder CLIP model.
        hf_config = AutoConfig.from_pretrained(model_name).vision_config
        hf_config.architecture = 'vit_clip_vision_encoder'
        hf_config.num_classes = 'n/a'
    else:
        hf_config = AutoConfig.from_pretrained(model_name)
        
#     print('hf config', hf_config)
            
 
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
    
    # Rectifying Huggingface bugs:
    # Currently a bug getting configs, only this model confirmed to work and even it requires modification of eps
    if is_timm and model_name == "vit_base_patch16_224":
        pretrained_config.update({
            "eps": 1e-6,
            "return_type": "class_logits",
        })
    
    # Config for 32 is incorrect, fix manually 
    if is_timm and model_name == "vit_base_patch32_224":
        pretrained_config.update({
            "patch_size": 32,
            "eps": 1e-6,
            "return_type": "class_logits"
        })
    
    print(pretrained_config)

    return HookedViTConfig.from_dict(pretrained_config)
