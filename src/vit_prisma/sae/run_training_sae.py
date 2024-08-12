import wandb
import torch

from vit_prisma.sae.config import VisionModelSAERunnerConfig
from vit_prisma.sae.train_sae import VisionSAETrainer


cfg = VisionModelSAERunnerConfig()
# cfg.lr = wandb.config.learning_rate
cfg.expansion_factor = wandb.config.expansion_factor
# print("Config created with lr" + str(cfg.lr))

cfg.l1_coefficient = wandb.config.l1_coefficient

cfg.hook_point_layer = 9

# Manually call __post_init__ to recalculate dependent values
cfg.__post_init__()

print(f"Config created with lr: {cfg.lr}, expansion_factor: {cfg.expansion_factor}, d_sae: {cfg.d_sae}, l1_coefficient: {cfg.l1_coefficient}, 'hook_point_layer: {cfg.hook_point_layer}")

trainer = VisionSAETrainer(cfg)
sae = trainer.run()

