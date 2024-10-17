import random
from multiprocessing import freeze_support

import numpy as np
import torch

from experiments.imagenet.imagenet_config import IMAGENET_CONFIG
from experiments.utils.train_utils import DemoCallback
from vit_prisma.models.base_vit import HookedViT
from vit_prisma.training import trainer
from vit_prisma.utils.constants import DATA_DIR
from vit_prisma.utils.data_utils.cifar.cifar_10_utils import load_imagenet

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


def train_imagenet(cfg):
    train_data, val_data, _ = load_imagenet(
        dataset_path=str(DATA_DIR / "imagenet"),
        dataset_train_path=str(DATA_DIR / "imagenet/ILSVRC/Data/CLS-LOC/train"),
        dataset_val_path=str(DATA_DIR / "imagenet/ILSVRC/Data/CLS-LOC/val"),
        augmentation=True,
        image_size=cfg.image_size,
        with_index=True,
        visualisation=False,
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

    train_imagenet(IMAGENET_CONFIG)
