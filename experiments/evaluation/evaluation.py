from pathlib import Path
from typing import Tuple

import click
import einops
import torch
import torchvision
from tqdm import tqdm
from vit_prisma.models.base_vit import HookedViT
from vit_prisma.sae.evals.eval_neuron_basis import EvalConfig
from vit_prisma.sae.sae import SparseAutoencoder
from vit_prisma.transforms.open_clip_transforms import get_clip_val_transforms
from huggingface_hub import hf_hub_download
from vit_prisma.utils.load_model import load_model


class ImageNetValidationDataset(torchvision.datasets.ImageNet):

    def __init__(self, *args, **kwargs):
        self.return_index = kwargs.get("return_index", False)
        kwargs.pop("return_index", None)
        super().__init__(
            *args,
            **kwargs,
        )

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        if self.return_index:
            return img, target, index
        return img, target


@torch.no_grad()
def get_sae_features(x, model, sparse_autoencoder, cfg):
    x = x.to(cfg.device)
    _, c = model.run_with_cache(x)
    sae_in = c[cfg.hook_point]
    _, feature_acts, *_ = sparse_autoencoder(sae_in)
    return feature_acts


def get_top_k_samples(
    data_loader: torch.utils.data.DataLoader,
    model: HookedViT,
    sparse_autoencoder: SparseAutoencoder,
    cfg: EvalConfig,
    top_k_samples: int,
):
    d_sae = sparse_autoencoder.d_sae
    n_tokens = sparse_autoencoder.cfg.context_size
    top_values = torch.zeros((d_sae, top_k_samples), device=cfg.device)
    top_indices = torch.zeros((d_sae, top_k_samples), device=cfg.device)
    top_localization = torch.zeros((d_sae, top_k_samples, n_tokens), device=cfg.device)

    for x, _, i in tqdm(data_loader):
        i = i.to(cfg.device)
        i = i.repeat(d_sae, 1)

        feature_acts = get_sae_features(
            x, model, sparse_autoencoder, cfg
        )  # (topk, n_tokens, d_sae)
        feature_acts = einops.rearrange(
            feature_acts, "topk n_tokens d_sae -> d_sae topk n_tokens"
        )
        feature_acts_max = feature_acts.max(dim=-1).values  # (d_sae, n_tokens)

        concated_localizations = torch.cat([top_localization, feature_acts], dim=1)
        concated_values = torch.cat([top_values, feature_acts_max], dim=1)
        concated_indices = torch.cat([top_indices, i], dim=1)

        top_values, idx = torch.topk(
            concated_values, top_k_samples, dim=1, largest=True, sorted=True
        )
        top_indices = torch.gather(concated_indices, 1, idx)

        idx_expanded = einops.repeat(
            idx, "d_sae top_k -> d_sae top_k n_tokens", n_tokens=n_tokens
        )
        top_localization = torch.gather(
            concated_localizations, dim=1, index=idx_expanded
        )

    return top_values, top_indices, top_localization


def load_sae_and_model(
    repo_id="Prisma-Multimodal/8e32860c-clip-b-sae-gated-all-tokens-x64-layer-9-mlp-out-v1",
    file_name="n_images_2600058.pt",
    config_name="config.json",
    local_config=None,
) -> Tuple[SparseAutoencoder, HookedViT]:
    """
    Load and test SAE from HuggingFace
    """
    print(f"Loading model from {repo_id}...")

    sae_path = hf_hub_download(repo_id, file_name)
    config_path = hf_hub_download(repo_id, config_name)

    print(f"Model saved at: {sae_path}")

    sae = SparseAutoencoder.load_from_pretrained(
        sae_path, current_cfg=local_config
    )  # This now automatically gets config.json and converts into the VisionSAERunnerConfig object

    sae.cfg.sae_path = sae_path
    model = load_model(sae.cfg.model_class_name, sae.cfg.model_name)
    model = model.to(sae.cfg.device)

    return sae, model


@click.command()
@click.option(
    "-k", "--top-k-samples", default=20, help="Number of top samples to visualize"
)
@click.option(
    "--result-path", default="results/sonia/", help="Path to save the results"
)
@click.option(
    "--repo",
    default="Prisma-Multimodal/8e32860c-clip-b-sae-gated-all-tokens-x64-layer-9-mlp-out-v1",
    help="HuggingFace repo id",
)
def main(top_k_samples: int, result_path: str, repo: str):

    local_config = {
        "dataset_name": "imagenet1k",
        "dataset_path": "data/ImageNet-complete/",
        "dataset_train_path": "data/ImageNet-complete/train",
        "dataset_val_path": "data/ImageNet-complete/val",
    }

    # Load and test the model
    sparse_autoencoder, model = load_sae_and_model(
        local_config=local_config, repo_id=repo
    )
    cfg = sparse_autoencoder.cfg
    print(sparse_autoencoder.cfg.use_patches_only)

    result_path = Path(result_path)
    result_path.mkdir(parents=True, exist_ok=True)

    result_path = Path(result_path, f"top_{top_k_samples}_samples.pt")
    # 1. check if already calculated
    if result_path.exists():
        top_values, top_indices, top_localization = torch.load(result_path)
    # 2. if not, calculate and save
    else:
        data_transforms = get_clip_val_transforms(cfg.image_size)

        val_data = ImageNetValidationDataset(
            cfg.dataset_path,
            transform=data_transforms,
            split="val",
            return_index=True,
        )

        val_dataloader = torch.utils.data.DataLoader(
            val_data, batch_size=128, shuffle=True, num_workers=16
        )

        top_values, top_indices, top_localization = get_top_k_samples(
            val_dataloader, model, sparse_autoencoder, cfg, top_k_samples
        )

        torch.save((top_values, top_indices, top_localization), result_path)

    # 3. load and visualize

    print(top_values.shape, top_indices.shape, top_localization.shape)


if __name__ == "__main__":
    main()