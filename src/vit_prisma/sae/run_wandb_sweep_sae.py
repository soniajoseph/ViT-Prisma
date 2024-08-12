import wandb
import torch

from vit_prisma.sae.config import VisionModelSAERunnerConfig
from vit_prisma.sae.train_sae import VisionSAETrainer


def train():
    # Initialize wandb
    run = wandb.init()
    cfg = VisionModelSAERunnerConfig()

    cfg.expansion_factor = wandb.config.expansion_factor
    cfg.hook_point_layer = wandb.config.hook_point_layer

    # Hold these fixed
    cfg.l1_coefficient = 0.0001
    cfg.learning_rate = 0.001

    # Manually call __post_init__ to recalculate dependent values
    cfg.__post_init__()

    print(f"Config created with lr: {cfg.lr}, expansion_factor: {cfg.expansion_factor}, d_sae: {cfg.d_sae}, l1_coefficient: {cfg.l1_coefficient}")

    trainer = VisionSAETrainer(cfg)
    sae = trainer.run()

if __name__ == "__main__":
    train()