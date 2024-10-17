from typing import Tuple

import click
import torch

from experiments.cifar10.train_sae_cifar10 import CIFAR10_SAE_CONFIG
from vit_prisma.models.base_vit import HookedViT
from vit_prisma.sae.evals.evaluator import Evaluator
from vit_prisma.sae.sae import SparseAutoencoder
from vit_prisma.utils.constants import EvaluationContext, DATA_DIR
from vit_prisma.utils.data_utils.cifar.cifar_10_utils import load_mnist, load_cifar_10

from experiments.mnist.train_sae_mnist import MNIST_SAE_CONFIG


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


def load_local_sae_and_model(
    sae_cfg,
) -> Tuple[SparseAutoencoder, HookedViT]:
    """Load and test SAE from HuggingFace"""

    print(f"Loading SAE from {sae_cfg.sae_path}...")
    sae = SparseAutoencoder.load_from_pretrained(
        sae_cfg.sae_path, current_cfg=sae_cfg
    )
    print(f"The SAE: {sae}")

    model = HookedViT.from_local(sae_cfg.vit_model_cfg, sae_cfg.model_path).to(sae_cfg.device)
    model = model.to(sae.cfg.device)

    return sae, model

@click.command()
@click.option(
    "-s", "--sae-name", help="Name of the SAE file, in form "
     "`99184a8a-local-mnist-clean-expansion-16-layer-3`"
)
@click.option(
    "-m", "--sae-model", help="Name of the SAE model file, in form "
     "`n_images_480003.pt`"
)
def main(sae_name: str, sae_model: str):

    sae_cfg = CIFAR10_SAE_CONFIG
    sae_cfg.sae_path = sae_cfg.checkpoint_path + f"/{sae_name}/{sae_model}"

    sparse_autoencoder, model = load_local_sae_and_model(
        sae_cfg=sae_cfg,
    )

    train_data, val_data, test_data = load_cifar_10(DATA_DIR / "cifar", augmentation=False, visualisation=False, with_index=True)
    _, val_data_visualisation, test_data_visualisation = load_cifar_10(DATA_DIR / "cifar", augmentation=False, visualisation=True, with_index=True)

    evaluator = Evaluator(model, None, val_data, sae_cfg, visualize_data=val_data_visualisation)
    evaluator.evaluate(sparse_autoencoder, context=EvaluationContext.POST_TRAINING)


if __name__ == "__main__":
    # main("7276b323-local-cifar10-clean-expansion-64-layer-6", "n_images_120012.pt")
    main()

# f16def25-local-cifar10-clean-expansion-16-layer-6/n_images_400005.pt