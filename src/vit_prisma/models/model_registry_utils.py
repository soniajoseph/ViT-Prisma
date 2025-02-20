"""
Prisma Model Utilities
=====================
User-friendly functions for discovering and examining available models.
"""

from typing import List, Dict, Optional, Set, Union, Tuple
from tabulate import tabulate

from vit_prisma.utils.enums import ModelType
from vit_prisma.model_configs import (
    ModelCategory,
    MODEL_CATEGORIES,
    MODEL_CONFIGS,
    TEXT_SUPPORTED_MODELS,
    BASE_VISION_CONFIG,
    BASE_LARGE_VISION_CONFIG
)

def list_available_models(
    category: Optional[Union[ModelCategory, str]] = None,
    model_type: Optional[ModelType] = None,
    format: str = 'list'
) -> Union[List[str], str]:
    """
    List all available models, with optional filtering.
    
    Args:
        category: Filter by model category (e.g., ModelCategory.TIMM or 'timm')
        model_type: Filter by model type (VISION or TEXT)
        format: Output format ('list', 'tabular', or 'pretty')
        
    Returns:
        List of model names or formatted string representation
    """
    # Handle string category input
    if isinstance(category, str):
        try:
            category = ModelCategory(category.lower())
        except ValueError:
            valid_categories = [c.value for c in ModelCategory]
            raise ValueError(
                f"Invalid category: {category}. "
                f"Valid categories are: {valid_categories}"
            )
    
    # Filter by category if specified
    if category:
        models = [name for name, cat in MODEL_CATEGORIES.items() if cat == category]
    else:
        models = list(MODEL_CATEGORIES.keys())
        
    # Filter by model type if specified
    if model_type:
        if model_type == ModelType.VISION:
            # All models support vision
            pass
        elif model_type == ModelType.TEXT:
            models = [m for m in models if m in TEXT_SUPPORTED_MODELS]
        else:
            raise ValueError(f"Invalid model_type: {model_type}")
    
    # Apply requested format
    if format == 'list':
        return sorted(models)
    
    # Get additional info for tabular formats
    model_info = []
    for name in sorted(models):
        category = MODEL_CATEGORIES[name]
        supports_text = name in TEXT_SUPPORTED_MODELS
        vision_config = MODEL_CONFIGS[ModelType.VISION][name]
        
        model_info.append({
            'name': name,
            'category': category.value,
            'modalities': 'vision+text' if supports_text else 'vision',
            'layers': vision_config.get('n_layers', 'N/A'),
            'dim': vision_config.get('d_model', 'N/A'),
            'heads': vision_config.get('n_heads', 'N/A'),
            'patch_size': vision_config.get('patch_size', 'N/A'),
        })
    
    if format == 'tabular':
        # Return as list of dictionaries
        return model_info
    
    elif format == 'pretty':
        # Return formatted table
        headers = {
            'name': 'Model Name',
            'category': 'Category',
            'modalities': 'Modalities',
            'layers': 'Layers',
            'dim': 'Hidden Dim',
            'heads': 'Heads',
            'patch_size': 'Patch Size'
        }
        
        return tabulate(
            model_info,
            headers=headers,
            tablefmt='grid'
        )
    
    else:
        raise ValueError(f"Invalid format: {format}. Valid formats: 'list', 'tabular', 'pretty'")

def get_model_info(model_name: str) -> Dict:
    """
    Get detailed information about a specific model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Dictionary with model information
    """
    if model_name not in MODEL_CATEGORIES:
        raise ValueError(f"Model '{model_name}' is not registered")
    
    category = MODEL_CATEGORIES[model_name]
    vision_config = MODEL_CONFIGS[ModelType.VISION][model_name]
    supports_text = model_name in TEXT_SUPPORTED_MODELS
    
    info = {
        'name': model_name,
        'category': category.value,
        'modalities': ['vision'],
        'architecture': {
            'layers': vision_config.get('n_layers'),
            'hidden_dim': vision_config.get('d_model'),
            'mlp_dim': vision_config.get('d_mlp'),
            'attention_heads': vision_config.get('n_heads'),
            'head_dim': vision_config.get('d_head'),
        },
        'vision_config': {
            'patch_size': vision_config.get('patch_size'),
            'image_size': vision_config.get('image_size'),
            'cls_token': vision_config.get('use_cls_token', True),
            'layer_norm_pre': vision_config.get('layer_norm_pre', False),
            'output_type': vision_config.get('return_type', 'class_logits'),
            'output_dim': vision_config.get('n_classes'),
        }
    }
    
    # Add video-specific info if applicable
    if vision_config.get('is_video_transformer', False):
        info['video_config'] = {
            'tubelet_depth': vision_config.get('video_tubelet_depth'),
            'num_frames': vision_config.get('video_num_frames'),
        }
    
    # Add text info if applicable
    if supports_text:
        info['modalities'].append('text')
        text_config = MODEL_CONFIGS[ModelType.TEXT][model_name]
        info['text_config'] = {
            'vocab_size': text_config.get('vocab_size'),
            'context_length': text_config.get('context_length'),
            'layer_norm_pre': text_config.get('layer_norm_pre', True),
        }
    
    return info

def list_model_categories() -> List[str]:
    """List all available model categories."""
    return [category.value for category in ModelCategory]

def get_models_by_category(category: Union[ModelCategory, str]) -> List[str]:
    """
    Get all models belonging to a specific category.
    
    Args:
        category: Category to filter by
        
    Returns:
        List of model names
    """
    # Handle string input
    if isinstance(category, str):
        try:
            category = ModelCategory(category.lower())
        except ValueError:
            valid_categories = [c.value for c in ModelCategory]
            raise ValueError(
                f"Invalid category: {category}. "
                f"Valid categories are: {valid_categories}"
            )
    
    return [name for name, cat in MODEL_CATEGORIES.items() if cat == category]

def get_models_by_size(
    size: str,
    category: Optional[Union[ModelCategory, str]] = None
) -> List[str]:
    """
    Get models filtered by approximate size.
    
    Args:
        size: Size category ('small', 'base', 'large', 'huge')
        category: Optional category filter
        
    Returns:
        List of matching model names
    """
    valid_sizes = ['tiny', 'small', 'base', 'large', 'huge', 'giant']
    if size not in valid_sizes:
        raise ValueError(f"Invalid size: {size}. Valid sizes: {valid_sizes}")
    
    # Get initial model list, filtered by category if specified  
    if category:
        models = get_models_by_category(category)
    else:
        models = list(MODEL_CATEGORIES.keys())
    
    # Filter by size
    result = []
    for model in models:
        config = MODEL_CONFIGS[ModelType.VISION][model]
        
        # Determine size category based on parameters
        layers = config.get('n_layers', 12)
        dim = config.get('d_model', 768)
        
        model_size = 'base'  # Default
        
        if layers <= 6 or dim <= 384:
            model_size = 'tiny'
        elif layers <= 12 and dim <= 512:
            model_size = 'small'
        elif layers <= 12 and dim <= 768:
            model_size = 'base'
        elif layers <= 24 and dim <= 1024:
            model_size = 'large'
        elif layers <= 32 and dim <= 1280:
            model_size = 'huge'
        else:
            model_size = 'giant'
        
        # Add if it matches the requested size
        if model_size == size:
            result.append(model)
    
    return sorted(result)
