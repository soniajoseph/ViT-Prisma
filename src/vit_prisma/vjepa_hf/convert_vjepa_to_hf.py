# VJEPA to HF checkpoint conversion script
## Following DinoV2 https://github.com/huggingface/transformers/blob/main/src/transformers/models/dinov2/convert_dinov2_to_hf.py
import fire
import torch
import torch.nn as nn

from PIL import Image
import requests

from app.vjepa_hf.modeling_vjepa import VJEPAModel, VJEPAConfig, VJEPAImageProcessor
from app.vjepa_hf.configs import CONFIGS


import src.models.vision_transformer as vit
import app.vjepa_batched.models.vision_transformer as vit_batched


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    return image


@torch.no_grad()
def convert_vjepa_v1(model_name, ckpt_loc, outp_loc):
    og_m = torch.load(ckpt_loc)
    # copy encoder keys
    og_state_dict = og_m["encoder"]
    og_keys = og_state_dict.keys()
    config = CONFIGS["v1"][model_name]
    model = VJEPAModel(config).eval()
    state_dict = model.state_dict()
    emb_dim = state_dict["embeddings.position_embeddings"].size(-1)

    for key, val in og_state_dict.copy().items():
        val = og_state_dict.pop(key)
        key = key.replace("module.backbone.", "")
        if key.startswith("blocks."):
            key = key.replace("blocks.", "encoder.layer.")
        if "attn." in key:
            key = key.replace("attn.", "attention.")
        if key == "pos_embed":
            key = "embeddings.position_embeddings"
        if "patch_embed." in key:
            key = key.replace("patch_embed.", "embeddings.patch_embeddings.")
        if key.startswith("norm."):
            key = key.replace("norm.", "layernorm.")
        if "qkv." in key:
            prefix, suffix = key.split("qkv")
            if "bias" in suffix:
                q_e, k_e, v_e = val[0:emb_dim], val[emb_dim : emb_dim * 2], val[emb_dim * 2 :]
            else:
                q_e, k_e, v_e = val[0:emb_dim, :], val[emb_dim : emb_dim * 2, :], val[emb_dim * 2 :, :]
            og_state_dict[prefix + "query" + suffix] = q_e
            og_state_dict[prefix + "key" + suffix] = k_e
            og_state_dict[prefix + "value" + suffix] = v_e
        else:
            og_state_dict[key] = val

    model.load_state_dict(og_state_dict)
    model.save_pretrained(outp_loc)
    del model


@torch.no_grad()
def test_vjepa_v1(model_name, og_loc, conv_loc):
    # test model load
    config = CONFIGS["v1"][model_name]
    model = VJEPAModel.from_pretrained(conv_loc)
    og_model = vit.__dict__[config.model_name](
        input_size=(config.crop_size, config.crop_size),
        patch_size=config.patch_size,
        num_frames=config.frames_per_clip,
        tubelet_size=config.tubelet_size,
        uniform_power=config.uniform_power,
        use_sdpa=config.use_sdpa,
        use_SiLU=config.use_SiLU,
        wide_SiLU=config.wide_SiLU,
        ignore_init=True,
    )

    og_state_dict = torch.load(og_loc)["encoder"]
    for key, val in og_state_dict.copy().items():
        val = og_state_dict.pop(key)
        key = key.replace("module.backbone.", "")
        og_state_dict[key] = val

    og_model.load_state_dict(og_state_dict)

    image = prepare_img()
    processor = VJEPAImageProcessor()
    pixel_values = processor(image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.repeat(1, 16, 1, 1, 1)
    pixel_values = pixel_values.permute(0, 2, 1, 3, 4)  # B x C x T x H x W
    with torch.no_grad():
        original_outputs = og_model(pixel_values)
        outputs = model(pixel_values)
        assert torch.allclose(outputs.last_hidden_state, original_outputs, atol=1e-3)
    print("Checks complete!")


@torch.no_grad()
def convert_vjepa_v1_5(model_name, ckpt_loc, outp_loc):
    # ckpt_loc = "/checkpoint/amaia/video/koustuvs/models/vjepa_v1.5/vm2myt1bin1k-16fpc-vitg16-256px-local-ropev2/latest.pth.tar"
    # model_name = "vit_g_256"
    og_m = torch.load(ckpt_loc)
    # copy encoder keys
    og_state_dict = og_m["encoder"]
    og_keys = og_state_dict.keys()
    config = CONFIGS["v1.5"][model_name]
    model = VJEPAModel(config).eval()
    state_dict = model.state_dict()
    emb_dim = state_dict["embeddings.position_embeddings"].size(-1)

    for key, val in og_state_dict.copy().items():
        val = og_state_dict.pop(key)
        key = key.replace("module.backbone.", "")
        if key.startswith("blocks."):
            key = key.replace("blocks.", "encoder.layer.")
        if "attn." in key:
            key = key.replace("attn.", "attention.")
        if key == "pos_embed":
            key = "embeddings.position_embeddings"
        if "patch_embed." in key:
            key = key.replace("patch_embed.", "embeddings.patch_embeddings.")
        if key.startswith("norm."):
            key = key.replace("norm.", "layernorm.")
        if "qkv." in key:
            prefix, suffix = key.split("qkv")
            if "bias" in suffix:
                q_e, k_e, v_e = val[0:emb_dim], val[emb_dim : emb_dim * 2], val[emb_dim * 2 :]
            else:
                q_e, k_e, v_e = val[0:emb_dim, :], val[emb_dim : emb_dim * 2, :], val[emb_dim * 2 :, :]
            og_state_dict[prefix + "query" + suffix] = q_e
            og_state_dict[prefix + "key" + suffix] = k_e
            og_state_dict[prefix + "value" + suffix] = v_e
        else:
            og_state_dict[key] = val

    model.load_state_dict(og_state_dict)
    model.save_pretrained(outp_loc)
    del model


@torch.no_grad()
def test_vjepa_v1_5(model_name, og_loc, conv_loc):
    # test model load
    config = CONFIGS["v1.5"][model_name]
    model = VJEPAModel.from_pretrained(conv_loc)
    og_model = vit_batched.__dict__[config.model_name](
        img_size=(config.crop_size, config.crop_size),
        patch_size=config.patch_size,
        num_frames=config.frames_per_clip,
        tubelet_size=config.tubelet_size,
        uniform_power=config.uniform_power,
        use_sdpa=config.use_sdpa,
        use_SiLU=config.use_SiLU,
        wide_SiLU=config.wide_SiLU,
        local_window=(-1, -1, -1),
        ignore_init=True,
    )

    og_state_dict = torch.load(og_loc)["encoder"]
    for key, val in og_state_dict.copy().items():
        val = og_state_dict.pop(key)
        key = key.replace("module.backbone.", "")
        og_state_dict[key] = val

    og_model.load_state_dict(og_state_dict)

    image = prepare_img()
    processor = VJEPAImageProcessor(crop_size=config.crop_size)
    pixel_values = processor(image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.repeat(1, 16, 1, 1, 1)
    pixel_values = pixel_values.permute(0, 2, 1, 3, 4)  # B x C x T x H x W
    with torch.no_grad():
        original_outputs = og_model(pixel_values)
        outputs = model(pixel_values)
        assert torch.allclose(outputs.last_hidden_state, original_outputs, atol=1e-3)
    print("Checks complete!")


def main(model_name, ckpt_loc, outp_loc, model_type="v1", convert=False, test=False):
    if model_type == "v1":
        convert_fn = convert_vjepa_v1
        test_fn = test_vjepa_v1
    elif model_type == "v1.5":
        convert_fn = convert_vjepa_v1_5
        test_fn = test_vjepa_v1_5
    else:
        raise NotImplementedError(f"model type {model_type} not implemented.")
    if convert:
        convert_fn(model_name, ckpt_loc, outp_loc)
    if test:
        test_fn(model_name, ckpt_loc, outp_loc)


if __name__ == "__main__":
    fire.Fire(main)
