from enum import Enum

import torch
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
MODEL_CHECKPOINTS_DIR = MODEL_DIR / "checkpoints"
SAE_CHECKPOINTS_DIR = MODEL_DIR / "sae"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
