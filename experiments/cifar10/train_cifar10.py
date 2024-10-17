import random
from multiprocessing import freeze_support

import numpy as np
import torch
from vit_prisma.models.base_vit import HookedViT
from vit_prisma.training import trainer
from vit_prisma.utils.constants import MODEL_CHECKPOINTS_DIR, DATA_DIR
from vit_prisma.utils.data_utils.cifar.cifar_10_utils import load_cifar_10

from experiments.cifar10.cifar10_config import CIFAR10_CONFIG
from experiments.utils.train_utils import DemoCallback

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


def train_cifar10(cfg):
    train_data, val_data, _ = load_cifar_10(
        DATA_DIR / "cifar", augmentation=True, visualisation=False, image_size=cfg.image_size,
    )

    model_function = HookedViT
    trainer.train(
        model_function,
        cfg,
        train_dataset=train_data,
        val_dataset=val_data,
        callbacks=[DemoCallback()],
    )


if __name__ == "__main__":
    freeze_support()

    train_cifar10(CIFAR10_CONFIG)
