import wandb
import torch

from vit_prisma.sae.config import VisionModelSAERunnerConfig
from vit_prisma.sae.train_sae import VisionSAETrainer

  
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
            # 'l1_coefficient': {
            #     'values': [1e-5, 5e-5, 1e-4, 0.0002, 0.0003, 0.0005, 0.001, 0.01, 0.1]
            # },
            'expansion_factor': {
                'values': [16, 32, 64, 128]
            },
            'hook_point_layer': {
                'values': list(range(12))  # 0 to 11 inclusive
            }
        },
        'program': 'run_wandb_sweep_sae.py',
    }
    # create sweep
    sweep_id = wandb.sweep(sweep_configuration, project="tinyclip40M_sae_sweep_learning_rate_all_layers")
    print("Sweep created with id: " + sweep_id)

    return 



if __name__ == "__main__":

    create_sweep()

