import click
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from vit_prisma.models.base_vit import HookedViT
from vit_prisma.training import trainer
from vit_prisma.utils.constants import MODEL_CHECKPOINTS_DIR, DATA_DIR
from vit_prisma.utils.data_utils.cifar.cifar_10_utils import load_cifar_10

from experiments.attack.PGD import L2PGDAttack, LinfPGDAttack
from experiments.cifar10.cifar10_config import CIFAR10_CONFIG
from experiments.utils.train_utils import DemoCallback


@click.command()
@click.option(
    "-s", "--sae-path", help="Path to the SAE path"
)
@click.option(
    "-n", "--norm", help="Which type of norm for the adversary"
)
@click.option(
    "-e", "--epsilon", default=0.5,
)
@click.option(
    "-a", "--alpha", default=0.1,
)
@click.option(
    "-k", "--attack-num-iters",  default=10,
)
def main(norm: str, epsilon: float, alpha: float, attack_num_iters: int):

    model_path = str(MODEL_CHECKPOINTS_DIR / "cifar10/clean/model_4090960.pth")
    model_cfg = CIFAR10_CONFIG
    model_cfg.attack_method = norm
    model_cfg.attack_epsilon = epsilon
    model_cfg.attack_alpha = alpha
    model_cfg.attack_num_iters = attack_num_iters
    # From their runs: epoch = 200, lr=0.1, k=10, eps=0.5, alpha=0.1

    model = HookedViT.from_local(model_cfg, model_path).to(model_cfg.device)
    # model = torch.nn.DataParallel(model)  # TODO EdS: Implement multi-GPU training

    cudnn.benchmark = True

    if norm == 'l2':
        adversary = L2PGDAttack(model, epsilon, alpha, k)
    elif norm == 'linf':
        adversary = LinfPGDAttack(model, epsilon, alpha, k)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=model_cfg.weight_decay)

    train_data, val_data, _ = load_cifar_10(DATA_DIR / "cifar", augmentation=True)

    CIFAR10_CONFIG.save_dir = str(MODEL_CHECKPOINTS_DIR / "cifar10-clean")
    model_function = HookedViT
    model = trainer.train(
        model_function,
        CIFAR10_CONFIG,
        train_dataset=train_data,
        val_dataset=val_data,
        callbacks=[DemoCallback()],
    )
