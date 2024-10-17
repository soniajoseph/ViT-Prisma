import inspect
from dataclasses import fields

import torch
import wandb

from experiments.cifar100.cifar100_config import CIFAR100_CONIG
from vit_prisma.configs.HookedViTConfig import HookedViTConfig
from vit_prisma.models.base_vit import HookedViT
from vit_prisma.training import trainer
from vit_prisma.training.training_utils import PrismaCallback
from vit_prisma.utils.constants import MODEL_CHECKPOINTS_DIR, DATA_DIR
from vit_prisma.utils.data_utils.cifar.cifar_10_utils import load_cifar_100


def train():
    # Initialize wandb
    run = wandb.init()

    # Create default config
    default_cfg = HookedViTConfig()

    # Create the actual config we'll use
    cfg = CIFAR100_CONIG
    cfg.model_name = "imagenet100-clean"
    cfg.save_dir = str(MODEL_CHECKPOINTS_DIR / "cifar100/clean")

    # Automatically detect and apply sweep parameters
    for key, value in wandb.config.items():
        if hasattr(cfg, key) and getattr(default_cfg, key) != value:
            setattr(cfg, key, value)
            print(f"Sweep parameter detected: {key} = {value}")

    # Apply fixed parameters
    fixed_params = {}
    for param, value in fixed_params.items():
        setattr(cfg, param, value)
        print(f"Fixed parameter applied: {param} = {value}")

    # Explicitly check and set the learning rate
    if 'lr' in wandb.config:
        cfg.lr = wandb.config.lr
        print(f"Setting learning rate to: {cfg.lr}")

    # Manually call __post_init__ to recalculate dependent values
    cfg.__post_init__()

    # Log the full configuration
    full_config = {}
    for field in fields(cfg):
        value = getattr(cfg, field.name)
        if isinstance(value, torch.dtype):
            value = str(value).split('.')[-1]  # Convert torch.dtype to string
        elif isinstance(value, torch.device):
            value = str(value)  # Convert torch.device to string
        elif callable(value) or inspect.isclass(value):
            continue  # Skip methods and classes
        full_config[field.name] = value

    wandb.config.update(full_config)

    # print full_config
    print("Configuration:")
    print(full_config)

    train_dataset, val_dataset, test_dataset = load_cifar_100(
        DATA_DIR / "cifar",
        image_size=cfg.image_size,
        augmentation=True,
    )

    class DemoCallback(PrismaCallback):
        def on_epoch_end(self, epoch, net, val_loader, wandb_logger):
            print(f"You reached the end of epoch: {epoch}")

        def on_step_end(self, step, net, val_loader, wandb_logger):
            if step % 200 == 0:
                print(f"You've reached step: {step}")

    model_function = HookedViT
    trainer.train(
        model_function,
        cfg,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        callbacks=[DemoCallback()],
    )


if __name__ == "__main__":
    train()