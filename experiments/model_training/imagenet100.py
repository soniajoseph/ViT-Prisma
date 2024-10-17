import random
from multiprocessing import freeze_support

import random
from multiprocessing import freeze_support

import numpy as np
import torch

from experiments.model_training.imagenet100_config import IMAGENET100_VIT_CONFIG_FAST
from vit_prisma.models.base_vit import HookedViT
from vit_prisma.training import trainer
from vit_prisma.training.training_utils import PrismaCallback
from vit_prisma.utils.constants import MODEL_CHECKPOINTS_DIR
from vit_prisma.utils.data_utils.cifar.cifar_10_utils import load_imagenet100

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


if __name__ == "__main__":
    freeze_support()

    IMAGENET100_VIT_CONFIG_FAST.model_name = "imagenet100-clean"
    IMAGENET100_VIT_CONFIG_FAST.save_dir = str(MODEL_CHECKPOINTS_DIR / "imagenet100-fast/clean")
    train_dataset, val_dataset = load_imagenet100(IMAGENET100_VIT_CONFIG_FAST.image_size)

    class DemoCallback(PrismaCallback):
        def on_epoch_end(self, epoch, net, val_loader, wandb_logger):
            print(f"You reached the end of epoch: {epoch}")

        def on_step_end(self, step, net, val_loader, wandb_logger):
            if step % 200 == 0:
                print(f"You've reached step: {step}")

    # check_dataset_stats(train_dataset)

    model_function = HookedViT
    model = trainer.train(
        model_function,
        IMAGENET100_VIT_CONFIG_FAST,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        callbacks=[DemoCallback()],
    )
