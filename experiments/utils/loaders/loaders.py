from typing import Tuple

from vit_prisma.models.base_text_transformer import HookedTextTransformer
from vit_prisma.models.base_vit import HookedViT
from vit_prisma.models.open_clip_models import hf_hub_download
from vit_prisma.sae.sae import SparseAutoencoder
from vit_prisma.utils.constants import DEVICE


def load_remote_sae_and_model(
    repo_id: str,
    checkpoint="n_images_2600058.pt",
    config_file: str = "config.json",
    current_cfg: dict = None
) -> Tuple[SparseAutoencoder, HookedTextTransformer, HookedViT]:
    """Load and test SAE from HuggingFace."""

    print(f"Loading SAE from repo_id: {repo_id} with checkpoint: {checkpoint}")
    sae_path = hf_hub_download(repo_id, checkpoint)
    sae_config_path = hf_hub_download(repo_id, config_file)

    sae = SparseAutoencoder.load_from_pretrained(sae_path, config_path=sae_config_path, current_cfg=current_cfg)

    print(f"Loading model name: {sae.cfg.model_name}")
    print(f"The config device is: {sae.cfg.device}")
    language_model = HookedTextTransformer.from_pretrained(sae.cfg.model_name, is_timm=False, is_clip=True).to(sae.cfg.device)
    model = HookedViT.from_pretrained(sae.cfg.model_name, is_timm=False, is_clip=True).to(sae.cfg.device)

    sae = sae.to(DEVICE)
    print(f"Using device: {DEVICE}")

    return sae, language_model, model