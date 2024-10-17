import os

import datasets
from torchvision import transforms
from pathlib import Path
from typing import Tuple

from experiments.utils.loaders.loaders import load_remote_sae_and_model
# from experiments.utils.performance_utils import calculate_accuracy
from experiments.utils.visualise import plot_image

import click
import einops
import torch
import torchvision
from tqdm import tqdm
from vit_prisma.models.base_vit import HookedViT
from vit_prisma.sae.evals.eval_neuron_basis import EvalConfig
from vit_prisma.sae.evals.evaluator import Evaluator
from vit_prisma.sae.sae import SparseAutoencoder
from vit_prisma.utils.constants import EvaluationContext, DATA_DIR, MODEL_DIR
from vit_prisma.utils.data_utils.cifar.cifar_10_utils import load_cifar_10

from experiments.cifar10.train_sae_cifar10 import CIFAR10_SAE_CONFIG
from experiments.utils.visualise import display_grid_on_image
from vit_prisma.utils.data_utils.loader import load_dataset


class Cifar10DatasetWithIndex(torch.utils.data.Subset):
    def __getitem__(self, current_index: int):
        return *super().__getitem__(current_index), current_index


class Cifar10DatasetWithIndices(torch.utils.data.Subset):
    def __getitem__(self, current_index: int):
        return *super().__getitem__(current_index), current_index, self.indices[current_index]


@torch.no_grad()
def get_sae_features(x, model, sparse_autoencoder, cfg):
    x = x.to(cfg.device)
    _, c = model.run_with_cache(x)
    sae_in = c[cfg.hook_point]
    _, feature_acts, *_ = sparse_autoencoder(sae_in)
    return feature_acts


def get_top_k_samples(  # TODO EdS:  This is for patches
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
        feature_acts_max = feature_acts.max(dim=-1).values  # (d_sae, n_tokens) - i.e. returns the highest activating patch in image

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
    sae_cfg,
) -> Tuple[SparseAutoencoder, HookedViT]:
    """Load and test SAE from HuggingFace"""

    print(f"Loading SAE from {sae_cfg.sae_path}...")
    sae = SparseAutoencoder.load_from_pretrained(
        sae_cfg.sae_path, current_cfg=sae_cfg
    )

    model = HookedViT.from_local(sae_cfg.vit_model_cfg, sae_cfg.model_path).to(sae_cfg.device)
    model = model.to(sae.cfg.device)

    return sae, model


def manually_evaluate_features(values, indices, localization, dataset):
    feature_idx = 10

    for i in range(feature_idx):
        img_indices = indices[feature_idx][i]
        img, label, idx = dataset[feature_idx, i]
        display_grid_on_image(img)
        dataset[indices[i]]


@click.command()
@click.option(
    "-k", "--top-k-samples", default=20, help="Number of top samples to visualize"
)
@click.option(
    "-r", "--result-path", help="Path to save the results"
)
def main(top_k_samples: int, result_path: str):
    repo_id = "Prisma-Multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-8-hook_mlp_out-l1-1e-05"
    # repo_id = "Prisma-Multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-8-hook_mlp_out-l1-0.0001"
    # "Prisma-Multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-8-hook_mlp_out-l1-0.0001": EvalStats(
    #   avg_loss=0.05401738430976868,
    #   avg_cos_sim=np.float64(0.9087297875253136),
    #   avg_reconstruction_loss=0.05401738603591919,
    #   avg_zero_abl_loss=0.054017670669555665,
    #   avg_l0=np.float32(98.14536),
    #   avg_l0_cls=np.float32(61.12024),
    #   avg_l0_image=np.float32(4870.2427),
    #   log_frequencies_per_token=array([-5.2839966, -5.2839966, -5.09691  , ..., -1.8874352, -5.251812 ,
    #        -5.1674914], dtype=float32),
    #   log_frequencies_per_image=array([-3.585027  , -3.585027  , -3.3979402 , ..., -0.18846522,
    #        -3.5528421 , -3.4685214 ], dtype=float32)
    # )
    sae_path = str(MODEL_DIR / "sae/imagenet/checkpoints") + f"/{repo_id}"
    os.makedirs(sae_path, exist_ok=True)

    complete_evals = True

    overload_cfg = {
        "wandb_log_frequency": 100,
        "dataset_path": str(DATA_DIR / "imagenet"),
        "dataset_train_path": str(DATA_DIR / "imagenet/ILSVRC/Data/CLS-LOC/train"),
        "dataset_val_path": str(DATA_DIR / "imagenet/ILSVRC/Data/CLS-LOC/val"),
        "checkpoint_path": str(MODEL_DIR / "sae/imagenet/checkpoints"),
        "sae_path": sae_path,
        "wandb_entity": "Stevinson",
        "wandb_project": "imagenet",
        "log_to_wandb": False,
        "verbose": True,
    }

    sae, model = load_remote_sae_and_model(repo_id, current_cfg=overload_cfg)

    train, test_data, test_data_visualisation = load_dataset(sae.cfg, visualize=True)
    # test, test_data_visualisation = load_dataset(sae.cfg, visualize=True)

    # First check performance is what I expected
    # TODO EdS: test accuracy
    # print(f"Validation Accuracy: {calculate_accuracy(model, val_data)}")

    if not complete_evals:
        # result_path = Path(sae_cfg.checkpoint_path + f"/{result_path}")
        # result_path.mkdir(parents=True, exist_ok=True)
        # result_path = Path(result_path, f"top_{top_k_samples}_samples.pt")

        if result_path.exists():
            top_values, top_indices, top_localization = torch.load(result_path, map_location=torch.device('cpu'))
        else:
            val_dataloader = torch.utils.data.DataLoader(
                val_data, batch_size=128, shuffle=False, num_workers=1
            )

            top_values, top_indices, top_localization = get_top_k_samples(
                val_dataloader, model, sparse_autoencoder, sae_cfg, top_k_samples
            )

            print(f"Saving the results to: {result_path}")
            torch.save((top_values, top_indices, top_localization), result_path)

        manually_evaluate_features(top_values, top_indices, top_localization, val_data)
    else:
        evaluator = Evaluator(model, test_data, sae.cfg, visualize_data=test_data_visualisation)
        evaluator.evaluate(sae, context=EvaluationContext.POST_TRAINING)


if __name__ == "__main__":
    main()