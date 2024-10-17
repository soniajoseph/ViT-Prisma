import inspect
import pprint
from dataclasses import fields

import torch
import wandb

from experiments.cifar10.train_sae_cifar10 import CIFAR10_SAE_CONFIG
from vit_prisma.sae.config import VisionModelSAERunnerConfig
from vit_prisma.sae.train_sae import VisionSAETrainer


def train():
    # Initialize wandb
    run = wandb.init()

    # Create default config
    default_cfg = VisionModelSAERunnerConfig()

    # Create the actual config we'll use
    cfg = CIFAR10_SAE_CONFIG

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

    # Convert torch.dtype to string
    full_config["vit_model_cfg"].dtype = str(full_config["vit_model_cfg"].dtype).split('.')[-1]

    print("Configuration:")
    pprint.pprint(full_config)

    wandb.config.update(full_config)

    cfg = VisionModelSAERunnerConfig.from_dict(full_config)
    print(cfg)
    trainer = VisionSAETrainer(cfg)
    trainer.run()


if __name__ == "__main__":
    # TODO EdS: This can be merged with the train model scripts to run a wandb sweep
    train()