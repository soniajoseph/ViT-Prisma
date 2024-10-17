import random
from multiprocessing import freeze_support

import numpy as np
import torch

from experiments.cifar100.cifar100_config import CIFAR100_CONIG
from vit_prisma.models.base_vit import HookedViT
from vit_prisma.training import trainer
from vit_prisma.training.training_utils import PrismaCallback
from vit_prisma.utils.constants import MODEL_CHECKPOINTS_DIR, DATA_DIR
from vit_prisma.utils.data_utils.cifar.cifar_10_utils import load_cifar_100

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


if __name__ == "__main__":
    freeze_support()

    CIFAR100_CONIG.model_name = "imagenet100-clean"
    CIFAR100_CONIG.save_dir = str(MODEL_CHECKPOINTS_DIR / "cifar100/clean")
    train_dataset, val_dataset, test_dataset = load_cifar_100(
        DATA_DIR / "cifar",
        image_size=CIFAR100_CONIG.image_size,
        augmentation=True,
    )

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
        CIFAR100_CONIG,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        callbacks=[DemoCallback()],
    )
