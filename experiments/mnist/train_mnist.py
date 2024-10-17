from experiments.utils.visualise import plot_image
import random
from multiprocessing import freeze_support

import numpy as np
import torch
from vit_prisma.models.base_vit import HookedViT
from vit_prisma.training import trainer
from vit_prisma.utils.constants import MODEL_CHECKPOINTS_DIR, DATA_DIR
from vit_prisma.utils.data_utils.cifar.cifar_10_utils import load_mnist

from experiments.mnist.mnist_config import MNIST_CONFIG
from experiments.utils.train_utils import DemoCallback

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


if __name__ == "__main__":
    freeze_support()

    train_data, val_data, test_data = load_mnist(DATA_DIR / "mnist", augmentation=False, visualisation=False)
    _, _, test_data_visualisation = load_mnist(DATA_DIR / "mnist", augmentation=False, visualisation=True)

    # Inspect data
    # plot_image(val_data[0][0], dataset="mnist", save_path="./network.png")
    # plot_image(test_visualisation[0][0], dataset="mnist", save_path="./visualise.png")

    MNIST_CONFIG.save_dir = str(MODEL_CHECKPOINTS_DIR / "mnist-clean")
    model_function = HookedViT
    model = trainer.train(
        model_function,
        MNIST_CONFIG,
        train_dataset=train_data,
        val_dataset=val_data,
        callbacks=[DemoCallback()],
    )
