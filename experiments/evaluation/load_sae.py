import sys
import torch
from vit_prisma.sae.sae import SparseAutoencoder
from vit_prisma.sae.config import VisionModelSAERunnerConfig
import matplotlib.pyplot as plt

from huggingface_hub import hf_hub_download
import numpy as np


def load_and_test_sae(
        repo_id="Prisma-Multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-8-hook_mlp_out-l1-1e-05",
        checkpoint="n_images_2600058.pt"):
    """
    Load and test SAE from HuggingFace
    """
    print(f"Loading model from {repo_id}...")

    # Download config.json and  get path

    sae_path = hf_hub_download(repo_id, checkpoint)

    sae = SparseAutoencoder.load_from_pretrained(sae_path)

    print(sae)

    print(type(sae.cfg))

    # Move to available device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sae = sae.to(device)
    print(f"Using device: {device}")

    return sae


# Load and test the model
sae = load_and_test_sae()