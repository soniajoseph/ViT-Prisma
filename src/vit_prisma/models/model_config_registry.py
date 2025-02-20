"""
Prisma Model Configurations
==========================
Static configuration definitions for all supported models.
This module contains only data, no logic, for better maintainability.
"""

from typing import Dict, Any
from enum import Enum

from vit_prisma.utils.enums import ModelType

# Model categories enum
class ModelCategory(Enum):
    TIMM = "timm"
    CLIP = "clip"
    OPEN_CLIP = "open_clip"
    DINO = "dino"
    VIVIT = "vivit"
    VJEPA = "vjepa"
    KANDINSKY = "kandinsky"

# ===============================
# Vision Model Configurations
# ===============================

# Base vision configurations that can be extended
BASE_VISION_CONFIG = {
    # Common defaults
    "d_model": 768,
    "n_heads": 12,
    "n_layers": 12,
    "d_mlp": 3072,
    "d_head": 64,
    "n_classes": 1000,
    "patch_size": 16,
    "image_size": 224,
    "eps": 1e-6,
    "normalization_type": "LN",
    "use_cls_token": True,
    "return_type": "class_logits"
}

BASE_LARGE_VISION_CONFIG = {
    **BASE_VISION_CONFIG,
    "d_model": 1024,
    "n_heads": 16,
    "n_layers": 24,
    "d_mlp": 4096,
}

# TIMM model configurations
TIMM_CONFIGS = {
    "vit_base_patch16_224": {
        **BASE_VISION_CONFIG,
        "eps": 1e-6,
        "return_type": "class_logits",
    },
    "vit_base_patch32_224": {
        **BASE_VISION_CONFIG,
        "patch_size": 32,
        "eps": 1e-6,
        "return_type": "class_logits",
    },
    "vit_large_patch16_224": {
        **BASE_LARGE_VISION_CONFIG,
        "eps": 1e-6,
        "return_type": "class_logits",
    },
}

# CLIP model configurations
CLIP_CONFIGS = {
    "openai/clip-vit-base-patch16": {
        **BASE_VISION_CONFIG,
        "layer_norm_pre": True,
        "return_type": "class_logits",
    },
    "openai/clip-vit-base-patch32": {
        **BASE_VISION_CONFIG,
        "patch_size": 32,
        "layer_norm_pre": True,
        "return_type": "class_logits", 
    },
    "openai/clip-vit-large-patch14": {
        **BASE_LARGE_VISION_CONFIG,
        "patch_size": 14,
        "layer_norm_pre": True,
        "return_type": "class_logits",
    },
}

# OpenCLIP model configurations
OPEN_CLIP_CONFIGS = {
    "open-clip:laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K": {
        **BASE_VISION_CONFIG,
        "patch_size": 32,
        "layer_norm_pre": True,
        "return_type": "class_logits",
    },
    "open-clip:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K": {
        **BASE_LARGE_VISION_CONFIG,
        "patch_size": 14,
        "layer_norm_pre": True,
        "return_type": "class_logits",
    },
    "open-clip:laion/CLIP-ViT-B-32-xlm-roberta-base-laion5B-s13B-b90k": {
        **BASE_VISION_CONFIG,
        "patch_size": 32,
        "layer_norm_pre": True,
        "return_type": "class_logits",
    },
    "open-clip:laion/CLIP-ViT-B-32-roberta-base-laion2B-s12B-b32k": {
        **BASE_VISION_CONFIG,
        "patch_size": 32,
        "layer_norm_pre": True,
        "return_type": "class_logits",
    },
    "open-clip:laion/CLIP-ViT-H-14-frozen-xlm-roberta-large-laion5B-s13B-b90k": {
        **BASE_LARGE_VISION_CONFIG,
        "d_model": 1280,
        "patch_size": 14,
        "layer_norm_pre": True,
        "return_type": "class_logits",
    },
    "open-clip:laion/CoCa-ViT-B-32-laion2B-s13B-b90k": {
        **BASE_VISION_CONFIG,
        "patch_size": 32,
        "layer_norm_pre": True,
        "return_type": "class_logits",
    },
    "open-clip:laion/CoCa-ViT-L-14-laion2B-s13B-b90k": {
        **BASE_LARGE_VISION_CONFIG,
        "patch_size": 14,
        "layer_norm_pre": True,
        "return_type": "class_logits",
    },
}

# EVA02 model configurations (special cases in OpenCLIP)
EVA02_CONFIGS = {
    "open-clip:timm/eva02_base_patch16_clip_224.merged2b_s8b_b131k": {
        **BASE_VISION_CONFIG,
        "layer_norm_pre": True,
        "return_type": "class_logits",
    },
    "open-clip:timm/eva02_enormous_patch14_clip_224.laion2b_s4b_b115k": {
        "d_model": 1792,
        "n_heads": 16,
        "n_layers": 40,
        "d_mlp": 7168,
        "d_head": 112,
        "patch_size": 14,
        "image_size": 224,
        "n_classes": 1000,
        "layer_norm_pre": True,
        "return_type": "class_logits",
        "normalization_type": "LN",
        "use_cls_token": True,
        "eps": 1e-6,
    },
    "open-clip:timm/eva02_enormous_patch14_plus_clip_224.laion2b_s9b_b144": {
        "d_model": 1792,
        "n_heads": 16,
        "n_layers": 40,
        "d_mlp": 7168,
        "d_head": 112,
        "patch_size": 14,
        "image_size": 224, 
        "n_classes": 1000,
        "layer_norm_pre": True,
        "return_type": "class_logits",
        "normalization_type": "LN",
        "use_cls_token": True,
        "eps": 1e-6,
    },
    "open-clip:timm/eva02_large_patch14_clip_224.merged2b_s4b_b131k": {
        **BASE_LARGE_VISION_CONFIG,
        "n_layers": 40,
        "patch_size": 14,
        "layer_norm_pre": True,
        "return_type": "class_logits",
    },
    "open-clip:timm/eva02_large_patch14_clip_336.merged2b_s6b_b61k": {
        **BASE_LARGE_VISION_CONFIG,
        "n_layers": 40,
        "patch_size": 14,
        "image_size": 336,
        "layer_norm_pre": True,
        "return_type": "class_logits",
    },
    "open-clip:timm/eva_giant_patch14_clip_224.laion400m_s11b_b41k": {
        "d_model": 1408,
        "n_heads": 16,
        "n_layers": 40,
        "d_mlp": 5632,
        "d_head": 88,
        "patch_size": 14,
        "image_size": 224,
        "n_classes": 1000,
        "layer_norm_pre": True,
        "return_type": "class_logits",
        "normalization_type": "LN",
        "use_cls_token": True,
        "eps": 1e-6,
    },
    "open-clip:timm/eva_giant_patch14_plus_clip_224.merged2b_s11b_b114k": {
        "d_model": 1408,
        "n_heads": 16,
        "n_layers": 40,
        "d_mlp": 5632,
        "d_head": 88,
        "patch_size": 14,
        "image_size": 224,
        "n_classes": 1000,
        "layer_norm_pre": True,
        "return_type": "class_logits",
        "normalization_type": "LN",
        "use_cls_token": True,
        "eps": 1e-6,
    },
}

# DINO model configurations
DINO_CONFIGS = {
    "facebook/dino-vitb16": {
        **BASE_VISION_CONFIG,
        "return_type": "pre_logits",
        "n_classes": 768,
    },
    "facebook/dino-vitb8": {
        **BASE_VISION_CONFIG,
        "patch_size": 8,
        "return_type": "pre_logits", 
        "n_classes": 768,
    },
    "facebook/dino-vits16": {
        "d_model": 384,
        "n_heads": 6,
        "n_layers": 12,
        "d_mlp": 1536,
        "d_head": 64,
        "patch_size": 16,
        "image_size": 224,
        "return_type": "pre_logits",
        "n_classes": 384,
        "normalization_type": "LN",
        "use_cls_token": True,
        "eps": 1e-6,
    },
    "facebook/dino-vits8": {
        "d_model": 384,
        "n_heads": 6,
        "n_layers": 12,
        "d_mlp": 1536,
        "d_head": 64,
        "patch_size": 8,
        "image_size": 224,
        "return_type": "pre_logits",
        "n_classes": 384,
        "normalization_type": "LN",
        "use_cls_token": True,
        "eps": 1e-6,
    },
}

# ViViT model configurations
VIVIT_CONFIGS = {
    "google/vivit-b-16x2-kinetics400": {
        **BASE_VISION_CONFIG,
        "is_video_transformer": True,
        "video_tubelet_depth": 2,
        "video_num_frames": 16,
        "n_classes": 400,
        "return_type": "class_logits",
    },
    "google/vivit-l-16x2-kinetics400": {
        **BASE_LARGE_VISION_CONFIG,
        "is_video_transformer": True,
        "video_tubelet_depth": 2,
        "video_num_frames": 16,
        "n_classes": 400,
        "return_type": "class_logits",
    },
}

# VJEPA model configurations
VJEPA_CONFIGS = {
    "vjepa_v1_vit_huge": {
        "d_model": 1280,
        "n_heads": 16,
        "n_layers": 32,
        "d_mlp": 5120,
        "d_head": 80,
        "patch_size": 14,
        "image_size": 224,
        "n_classes": 1280,
        "use_cls_token": False,
        "layer_norm_pre": False,
        "return_type": "pre_logits",
        "classification_type": "last_hidden",
        "normalization_type": "LN",
        "eps": 1e-6,
    },
}

# Kandinsky model configurations
KANDINSKY_CONFIGS = {
    "kandinsky-community/kandinsky-2-1-prior": {
        "d_model": 1024,
        "n_heads": 16,
        "n_layers": 24,
        "d_mlp": 4096,
        "d_head": 64,
        "patch_size": 14,
        "image_size": 224,
        "n_classes": 768,
        "layer_norm_pre": True,
        "return_type": "class_logits",
        "normalization_type": "LN",
        "use_cls_token": True,
        "eps": 1e-5,
    },
}

# ===============================
# Text Model Configurations
# ===============================

# Base text configuration
BASE_TEXT_CONFIG = {
    "d_model": 768,
    "n_heads": 12,
    "n_layers": 12,
    "d_mlp": 3072,
    "d_head": 64,
    "vocab_size": 49408,
    "context_length": 77,
    "eps": 1e-5,
    "normalization_type": "LN",
    "layer_norm_pre": True,
}

# Text configurations for CLIP models
OPEN_CLIP_TEXT_CONFIGS = {
    # Add models that support text encoder
    "open-clip:laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K": {
        **BASE_TEXT_CONFIG,
    },
    "open-clip:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K": {
        **BASE_TEXT_CONFIG,
    },
}

# ===============================
# Combined Registry Dictionary
# ===============================

# Mapping model names to categories
MODEL_CATEGORIES = {
    **{name: ModelCategory.TIMM for name in TIMM_CONFIGS},
    **{name: ModelCategory.CLIP for name in CLIP_CONFIGS},
    **{name: ModelCategory.OPEN_CLIP for name in OPEN_CLIP_CONFIGS},
    **{name: ModelCategory.OPEN_CLIP for name in EVA02_CONFIGS},
    **{name: ModelCategory.DINO for name in DINO_CONFIGS},
    **{name: ModelCategory.VIVIT for name in VIVIT_CONFIGS},
    **{name: ModelCategory.VJEPA for name in VJEPA_CONFIGS},
    **{name: ModelCategory.KANDINSKY for name in KANDINSKY_CONFIGS},
}

# Combined configuration dictionary
MODEL_CONFIGS = {
    # Vision configurations
    ModelType.VISION: {
        **TIMM_CONFIGS,
        **CLIP_CONFIGS,
        **OPEN_CLIP_CONFIGS, 
        **EVA02_CONFIGS,
        **DINO_CONFIGS,
        **VIVIT_CONFIGS,
        **VJEPA_CONFIGS,
        **KANDINSKY_CONFIGS,
    },
    # Text configurations
    ModelType.TEXT: {
        **OPEN_CLIP_TEXT_CONFIGS,
    }
}

# Get the list of models that support text modality 
TEXT_SUPPORTED_MODELS = set(MODEL_CONFIGS[ModelType.TEXT].keys())