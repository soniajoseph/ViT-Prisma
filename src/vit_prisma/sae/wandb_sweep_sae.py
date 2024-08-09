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
            'learning_rate': {
                'values': [1e-4, 1.778e-4, 3.162e-4, 5.623e-4, 1e-3, 1.778e-3, 3.162e-3, 5.623e-3, 1e-2]
            },
            'expansion_factor': {
                'values': [32, 64, 128]
            }
        },
        'program': 'wandb_sweep_sae.py',
    }
    # create sweep
    sweep_id = wandb.sweep(sweep_configuration, project="sae_sweep")
    print("Sweep created with id: " + sweep_id)

    return 



if __name__ == "__main__":

    # create_sweep()

    train()
