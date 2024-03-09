"""
Reference:
https://github.com/neelnanda-io/TransformerLens/blob/main/transformer_lens/loading_from_pretrained.py

Preserves most of the original functionality with necessary modifications for ViTs and pretrained ViT models.
"""

import logging

from transformers import AutoConfig, ViTForImageClassification, VivitForVideoClassification, CLIPModel

import timm
from vit_prisma.configs.HookedViTConfig import HookedViTConfig

import torch

from typing import Dict

import einops




def convert_clip_weights(
        old_state_dict,
        old_head_state_dict,
        cfg: HookedViTConfig,
):
    
    new_vision_model_state_dict = {}


    new_vision_model_state_dict["cls_token"] = old_state_dict["embeddings.class_embedding"].unsqueeze(0).unsqueeze(0)
    new_vision_model_state_dict["pos_embed.W_pos"] = old_state_dict["embeddings.position_embedding.weight"]
    new_vision_model_state_dict["embed.proj.weight"] = old_state_dict["embeddings.patch_embedding.weight"]
    new_vision_model_state_dict["embed.proj.bias"] =  torch.zeros((cfg.d_model,), device=new_vision_model_state_dict["embed.proj.weight"].device)
    new_vision_model_state_dict["ln_final.w"] = old_state_dict["post_layernorm.weight"]
    new_vision_model_state_dict["ln_final.b"] = old_state_dict["post_layernorm.bias"]
    new_vision_model_state_dict["ln_pre.w"] = old_state_dict["pre_layrnorm.weight"] #typo in ClipModel
    new_vision_model_state_dict["ln_pre.b"] = old_state_dict["pre_layrnorm.bias"]


    for layer in range(cfg.n_layers):
        layer_key = f"encoder.layers.{layer}"
        new_layer_key = f"blocks.{layer}"

        new_vision_model_state_dict[f"{new_layer_key}.ln1.w"] = old_state_dict[f"{layer_key}.layer_norm1.weight"]
        new_vision_model_state_dict[f"{new_layer_key}.ln1.b"] = old_state_dict[f"{layer_key}.layer_norm1.bias"]
        new_vision_model_state_dict[f"{new_layer_key}.ln2.w"] = old_state_dict[f"{layer_key}.layer_norm2.weight"]
        new_vision_model_state_dict[f"{new_layer_key}.ln2.b"] = old_state_dict[f"{layer_key}.layer_norm2.bias"]

        W_Q = old_state_dict[f"{layer_key}.self_attn.q_proj.weight"]
        W_K = old_state_dict[f"{layer_key}.self_attn.k_proj.weight"]
        W_V = old_state_dict[f"{layer_key}.self_attn.v_proj.weight"]
        W_O = old_state_dict[f"{layer_key}.self_attn.out_proj.weight"]

        W_Q = einops.rearrange(W_Q, "(h dh) d-> h d dh", h=cfg.n_heads, d=cfg.d_model, dh=cfg.d_head)
        W_K = einops.rearrange(W_K, "(h dh) d-> h d dh", h=cfg.n_heads, d=cfg.d_model, dh=cfg.d_head)
        W_V = einops.rearrange(W_V, "(h dh) d-> h d dh", h=cfg.n_heads, d=cfg.d_model, dh=cfg.d_head)
        W_O = einops.rearrange(W_O, "d (h dh) -> h dh d", h=cfg.n_heads, d=cfg.d_model, dh=cfg.d_head)
        
        new_vision_model_state_dict[f"{new_layer_key}.attn.W_Q"] = W_Q
        new_vision_model_state_dict[f"{new_layer_key}.attn.W_K"] = W_K
        new_vision_model_state_dict[f"{new_layer_key}.attn.W_V"] = W_V
        new_vision_model_state_dict[f"{new_layer_key}.attn.W_O"] = W_O

        b_Q = old_state_dict[f"{layer_key}.self_attn.q_proj.bias"]
        b_K = old_state_dict[f"{layer_key}.self_attn.k_proj.bias"]
        b_V = old_state_dict[f"{layer_key}.self_attn.v_proj.bias"]
        b_O = old_state_dict[f"{layer_key}.self_attn.out_proj.bias"]

        b_Q = einops.rearrange(b_Q, "(h dh) -> h dh", h=cfg.n_heads, dh=cfg.d_head)
        b_K = einops.rearrange(b_K, "(h dh) -> h dh", h=cfg.n_heads, dh=cfg.d_head)
        b_V = einops.rearrange(b_V, "(h dh) -> h dh", h=cfg.n_heads, dh=cfg.d_head)

        new_vision_model_state_dict[f"{new_layer_key}.attn.b_Q"] = b_Q
        new_vision_model_state_dict[f"{new_layer_key}.attn.b_K"] = b_K
        new_vision_model_state_dict[f"{new_layer_key}.attn.b_V"] = b_V
        new_vision_model_state_dict[f"{new_layer_key}.attn.b_O"] = b_O

        mlp_W_in = old_state_dict[f"{layer_key}.mlp.fc1.weight"]
        mlp_W_out = old_state_dict[f"{layer_key}.mlp.fc2.weight"]
        mlp_b_in = old_state_dict[f"{layer_key}.mlp.fc1.bias"]
        mlp_b_out = old_state_dict[f"{layer_key}.mlp.fc2.bias"]

        mlp_W_in = einops.rearrange(mlp_W_in, "m d -> d m")
        mlp_W_out = einops.rearrange(mlp_W_out, "d m -> m d")

        new_vision_model_state_dict[f"{new_layer_key}.mlp.W_in"] = mlp_W_in
        new_vision_model_state_dict[f"{new_layer_key}.mlp.W_out"] = mlp_W_out
        new_vision_model_state_dict[f"{new_layer_key}.mlp.b_in"] = mlp_b_in
        new_vision_model_state_dict[f"{new_layer_key}.mlp.b_out"] = mlp_b_out

    new_vision_model_state_dict["head.W_H"] = einops.rearrange(old_head_state_dict["weight"], "c d -> d c")
    new_vision_model_state_dict["head.b_H"] = torch.zeros((cfg.n_classes,), device=new_vision_model_state_dict["head.W_H"].device)

        
    return new_vision_model_state_dict


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


def convert_vivet_weights(
        old_state_dict,
        cfg: HookedViTConfig,
):

    new_state_dict = {}

    new_state_dict["cls_token"] = old_state_dict["vivit.embeddings.cls_token"]
    new_state_dict["pos_embed.W_pos"] = old_state_dict["vivit.embeddings.position_embeddings"].squeeze(0)
    new_state_dict["embed.proj.weight"] = old_state_dict["vivit.embeddings.patch_embeddings.projection.weight"]
    new_state_dict["embed.proj.bias"] = old_state_dict["vivit.embeddings.patch_embeddings.projection.bias"] 
    new_state_dict["ln_final.w"] = old_state_dict["vivit.layernorm.weight"]
    new_state_dict["ln_final.b"] = old_state_dict["vivit.layernorm.bias"]

    for layer in range(cfg.n_layers):
        layer_key = f"vivit.encoder.layer.{layer}" 
        new_layer_key = f"blocks.{layer}"
        new_state_dict[f"{new_layer_key}.ln1.w"] = old_state_dict[f"{layer_key}.layernorm_before.weight"]
        new_state_dict[f"{new_layer_key}.ln1.b"] = old_state_dict[f"{layer_key}.layernorm_before.bias"]
        new_state_dict[f"{new_layer_key}.ln2.w"] = old_state_dict[f"{layer_key}.layernorm_after.weight"]
        new_state_dict[f"{new_layer_key}.ln2.b"] = old_state_dict[f"{layer_key}.layernorm_after.bias"]

        W_Q  = old_state_dict[f"{layer_key}.attention.attention.query.weight"]
        W_K  = old_state_dict[f"{layer_key}.attention.attention.key.weight"]
        W_V  = old_state_dict[f"{layer_key}.attention.attention.value.weight"]

        new_state_dict[f"{new_layer_key}.attn.W_Q"] = einops.rearrange(W_Q, "(h dh) d -> h d dh", h=cfg.n_heads, d=cfg.d_model, dh=cfg.d_head)
        new_state_dict[f"{new_layer_key}.attn.W_K"] = einops.rearrange(W_K, "(h dh) d -> h d dh", h=cfg.n_heads, d=cfg.d_model, dh=cfg.d_head)
        new_state_dict[f"{new_layer_key}.attn.W_V"] = einops.rearrange(W_V, "(h dh) d -> h d dh", h=cfg.n_heads, d=cfg.d_model, dh=cfg.d_head)

        W_O = old_state_dict[f"{layer_key}.attention.output.dense.weight"]
        new_state_dict[f"{new_layer_key}.attn.W_O"] =  einops.rearrange(W_O, "d (h dh) -> h dh d", h=cfg.n_heads, d=cfg.d_model, dh=cfg.d_head)


        b_Q  = old_state_dict[f"{layer_key}.attention.attention.query.bias"]
        b_K  = old_state_dict[f"{layer_key}.attention.attention.key.bias"]
        b_V  = old_state_dict[f"{layer_key}.attention.attention.value.bias"]

        new_state_dict[f"{new_layer_key}.attn.b_Q"] = einops.rearrange(b_Q, "(h dh) -> h dh",  h=cfg.n_heads, dh=cfg.d_head)
        new_state_dict[f"{new_layer_key}.attn.b_K"] = einops.rearrange(b_K, "(h dh) -> h dh",  h=cfg.n_heads, dh=cfg.d_head)
        new_state_dict[f"{new_layer_key}.attn.b_V"] = einops.rearrange(b_V, "(h dh) -> h dh",  h=cfg.n_heads, dh=cfg.d_head)

        b_O = old_state_dict[f"{layer_key}.attention.output.dense.bias"]
        new_state_dict[f"{new_layer_key}.attn.b_O"] = b_O

        mlp_W_in = old_state_dict[f"{layer_key}.intermediate.dense.weight"]
        new_state_dict[f"{new_layer_key}.mlp.W_in"] =  einops.rearrange(mlp_W_in, "m d -> d m")

        mlp_W_out  = old_state_dict[f"{layer_key}.output.dense.weight"]
       
        new_state_dict[f"{new_layer_key}.mlp.W_out"] = einops.rearrange(mlp_W_out, "d m -> m d")

        new_state_dict[f"{new_layer_key}.mlp.b_in"] =  old_state_dict[f"{layer_key}.intermediate.dense.bias"]
        new_state_dict[f"{new_layer_key}.mlp.b_out"] =  old_state_dict[f"{layer_key}.output.dense.bias"]


    new_state_dict["head.W_H"] = einops.rearrange(old_state_dict["classifier.weight"], "c d -> d c")
    new_state_dict["head.b_H"] = old_state_dict["classifier.bias"]


    return new_state_dict

def get_pretrained_state_dict(
    official_model_name: str,
    is_timm: bool,
    is_clip: bool,
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
            hf_model = hf_model if hf_model is not None else timm.create_model(official_model_name, pretrained=True)
            for param in hf_model.parameters():
                param.requires_grad = False
            state_dict = convert_timm_weights(hf_model.state_dict(), cfg)
        elif is_clip:
            full_model = hf_model if hf_model is not None else CLIPModel.from_pretrained(official_model_name)
            for param in full_model.parameters():
                param.requires_grad = False
            vision = full_model.vision_model
            visual_projection = full_model.visual_projection
            state_dict = convert_clip_weights(vision.state_dict(), visual_projection.state_dict(), cfg)

        elif cfg.is_video_transformer:
            if "vivit" in official_model_name:
                hf_model = hf_model if hf_model is not None else VivitForVideoClassification.from_pretrained(official_model_name, torch_dtype=dtype, **kwargs)

                for param in hf_model.parameters():
                    param.requires_grad = False

                state_dict = convert_vivet_weights(hf_model.state_dict(), cfg)
            else:
                raise ValueError
        
        else:
            hf_model = hf_model if hf_model is not None else ViTForImageClassification.from_pretrained(
                    official_model_name, torch_dtype=dtype, **kwargs
            )
            raise ValueError

            # Load model weights, and fold in layer norm weights


                
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
        hf_config.num_classes = hf_config.projection_dim # final output dimension instead of classes
    else:
        hf_config = AutoConfig.from_pretrained(model_name)
        
#     print('hf config', hf_config)
            
 

    if hasattr(hf_config, 'patch_size'):
        ps = hf_config.patch_size
    elif hasattr(hf_config, "tubelet_size"):
        ps = hf_config.tubelet_size[1]

    pretrained_config = {
                    'n_layers' : hf_config.num_hidden_layers,
                    'd_model' : hf_config.hidden_size,
                    'd_head' : hf_config.hidden_size // hf_config.num_attention_heads,
                    'model_name' : hf_config._name_or_path,
                    'n_heads' : hf_config.num_attention_heads,
                    'd_mlp' : hf_config.intermediate_size,
                    'activation_name' : hf_config.hidden_act,
                    'eps' : hf_config.layer_norm_eps,
                    'original_architecture' : getattr(hf_config, 'architecture', getattr(hf_config, 'architectures', None)),
                    'initializer_range' : hf_config.initializer_range,
                    'n_channels' : hf_config.num_channels,
                    'patch_size' : ps,
                    'image_size' : hf_config.image_size,
                    'n_classes' : getattr(hf_config, "num_classes", None),
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

    if is_clip:
        pretrained_config.update({
            "layer_norm_pre": True,
            "return_type": "class_logits" # actually returns 'visual_projection'
        })


    # Config is for ViVet, need to add more properties
    if hasattr(hf_config, "tubelet_size"):
        pretrained_config.update({
            "is_video_transformer": True,
            "video_tubelet_depth": hf_config.tubelet_size[0],
            "video_num_frames": hf_config.video_size[0],
            "n_classes": 400 if "kinetics400" in model_name else None,
            "return_type": "class_logits" if "kinetics400" in model_name else "pre_logits",

        })

    
    print(pretrained_config)

    return HookedViTConfig.from_dict(pretrained_config)
