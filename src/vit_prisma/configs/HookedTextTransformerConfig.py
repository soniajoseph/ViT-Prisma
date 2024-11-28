from dataclasses import dataclass
import torch.nn as nn
import torch

from typing import Any, Dict, List, Optional

from vit_prisma.configs.HookedViTConfig import HookedViTConfig


@dataclass
class HookedTextTransformerConfig(HookedViTConfig):
    """Config specific to the text transformer."""

    context_length: int = 77
    vocab_size: int = 10_000
