import pytest
import torch
import timm
from vit_prisma.models.base_vit import HookedViT
import open_clip

import numpy as np

import os

from vit_prisma.model_eval.evaluate_imagenet import zero_shot_eval
from vit_prisma.dataloaders.imagenet_dataset import load_imagenet

from transformers import CLIPVisionModelWithProjection

import torch

# For OpenCLIP models
cache_dir = '/network/scratch/s/sonia.joseph/hub'
os.environ['HUGGINGFACE_HUB_CACHE'] = cache_dir
os.environ['OPEN_CLIP_CACHE'] = cache_dir
os.environ['TRANSFORMERS_CACHE'] = cache_dir


random_input = torch.rand((1, 3, 224, 224))
hooked_vit_g = HookedViT.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", is_clip=True, is_timm=False, fold_ln=False)
hooked_vit_op = hooked_vit_g(random_input)


hf_model = CLIPVisionModelWithProjection.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", cache_dir=cache_dir)
hf_model.eval()
op = hf_model(random_input)

print(torch.allclose(hooked_vit_op, op.image_embeds, atol=1e-4))
