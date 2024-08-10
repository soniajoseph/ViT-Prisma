import wandb
import torch

from vit_prisma.sae.config import VisionModelSAERunnerConfig
from vit_prisma.sae.train_sae import VisionSAETrainer


def train():
    # Initialize wandb
    run = wandb.init()
    cfg = VisionModelSAERunnerConfig()
    cfg.lr = wandb.config.learning_rate
    cfg.expansion_factor = wandb.config.expansion_factor
    print("Config created with lr" + str(cfg.lr))

    # Manually call __post_init__ to recalculate dependent values
    cfg.__post_init__()

    print(f"Config created with lr: {cfg.lr}, expansion_factor: {cfg.expansion_factor}, d_sae: {cfg.d_sae}")

    trainer = VisionSAETrainer(cfg)
    sae = trainer.run()

def create_sweep():
    sweep_configuration = {
        'method': 'grid',
        'metric': {
            'name': 'overall_loss',
            'goal': 'minimize'
        },
        'parameters': {
            # 'learning_rate': {
            #     'values': [1e-4, 1.778e-4, 3.162e-4, 5.623e-4, 1e-3, 1.778e-3, 3.162e-3, 5.623e-3, 1e-2]
            # },
            'l1_coefficient': {
                'values': [1e-5, 5e-5, 1e-4, 0.0002, 0.0003, 0.0005, 0.001, 0.01, 0.1]
            },
            'expansion_factor': {
                'values': [16, 32, 64, 128]
            }
        },
        'program': 'wandb_sweep_sae.py',
    }
    # create sweep
    sweep_id = wandb.sweep(sweep_configuration, project="tinyclip40M_sae_sweep_l1_coefficient")
    print("Sweep created with id: " + sweep_id)

    return 



if __name__ == "__main__":

    create_sweep()

    # train()
