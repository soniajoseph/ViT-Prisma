import wandb
import torch
import inspect
from dataclasses import fields

from vit_prisma.sae.config import VisionModelSAERunnerConfig
from vit_prisma.sae.train_sae import VisionSAETrainer

def train():
    # Initialize wandb
    run = wandb.init()
    
    # Create default config
    default_cfg = VisionModelSAERunnerConfig()
    
    # Create the actual config we'll use
    cfg = VisionModelSAERunnerConfig()

    # Fixed parameters (manually set )
    fixed_params = {
        'expansion_factor': 32
    }

    # Automatically detect and apply sweep parameters
    for key, value in wandb.config.items():
        if hasattr(cfg, key) and getattr(default_cfg, key) != value:
            setattr(cfg, key, value)
            print(f"Sweep parameter detected: {key} = {value}")

    # Apply fixed parameters
    for param, value in fixed_params.items():
        setattr(cfg, param, value)
        print(f"Fixed parameter applied: {param} = {value}")

      #Explicitly check and set the learning rate
    if 'learning_rate' in wandb.config:
        cfg.lr = wandb.config.learning_rate
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

    trainer = VisionSAETrainer(cfg)
    sae = trainer.run()


if __name__ == "__main__":
    train()