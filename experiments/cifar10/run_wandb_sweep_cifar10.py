import inspect
from dataclasses import fields

import torch
import wandb

from experiments.cifar10.cifar10_config import CIFAR10_CONFIG
from experiments.cifar10.train_cifar10 import train_cifar10
from vit_prisma.configs.HookedViTConfig import HookedViTConfig


def train():
    # Initialize wandb
    run = wandb.init()

    # Create default config
    default_cfg = HookedViTConfig()

    # Create the actual config we'll use
    cfg = CIFAR10_CONFIG

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

    cfg = HookedViTConfig.from_dict(full_config)
    print(cfg)
    train_cifar10(cfg)


if __name__ == "__main__":
    train()