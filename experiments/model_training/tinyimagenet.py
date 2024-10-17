from torchvision import transforms
import random
from itertools import islice
from multiprocessing import freeze_support

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb

from experiments.model_training.tinyimagenet_config import IMAGENET_VIT_CONFIG
from experiments.utils.normalise import check_dataset_stats
from vit_prisma.utils.constants import MODEL_CHECKPOINTS_DIR
from vit_prisma.models.base_vit import HookedViT
from vit_prisma.training import trainer
from vit_prisma.training.training_utils import PrismaCallback
from vit_prisma.utils.constants import DEVICE
from vit_prisma.utils.data_utils.cifar.cifar_10_utils import load_tinyimagenet
from vit_prisma.utils.data_utils.loader import load_dataset

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

if __name__ == "__main__":
    freeze_support()

    train_dataset, val_dataset, test_dataset = load_tinyimagenet("tinyimagenet")

    class DemoCallback(PrismaCallback):
        def on_epoch_end(self, epoch, net, val_loader, wandb_logger):
            print(f"You reached the end of epoch: {epoch}")

        def on_step_end(self, step, net, val_loader, wandb_logger):
            if step % 200 == 0:
                net.eval()
                with torch.no_grad():
                    for items in islice(val_loader, 2000 // 256):
                        x, labels, *extras = items
                        logits = net(x.to(DEVICE))
                        loss = torch.nn.CrossEntropyLoss()(
                            logits, labels.to(DEVICE)
                        ).item()
                        print(f"Loss: {loss}")
                        print(f"Labels: {labels}")
                        print(f"Preds: {torch.argmax(logits, dim=-1)}")
                        print(f"logits: {logits}")
                        break


    # check_dataset_stats(train_dataset)

    IMAGENET_VIT_CONFIG.model_name = "tinyimagenet_clean"
    IMAGENET_VIT_CONFIG.save_dir = str(MODEL_CHECKPOINTS_DIR / "tinyimagenet-clean")
    model_function = HookedViT
    model = trainer.train(
        model_function,
        IMAGENET_VIT_CONFIG,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        callbacks=[DemoCallback()],
    )
