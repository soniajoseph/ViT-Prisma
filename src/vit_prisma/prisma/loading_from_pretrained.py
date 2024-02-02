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

def get_pretrained_state_dict(
    official_model_name: str,
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

            hf_model = ViTForImageClassification.from_pretrained(
                    official_model_name, torch_dtype=dtype, **kwargs
            )

            # Load model weights, and fold in layer norm weights

            for param in hf_model.parameters():
                param.requires_grad = False

            state_dict = None # Conversion of state dict to HookedTransformer format

            raise NotImplementedError(
                'Pending implementation of state dict conversion to HookedTransformer format.'
            )
                
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
        model = timm.create_model('vit_base_patch32_224')
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

    return HookedViTConfig.HookedViTConfig.from_dict(pretrained_config)
