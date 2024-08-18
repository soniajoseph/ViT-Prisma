import wandb
import torch
import argparse

from vit_prisma.sae.config import VisionModelSAERunnerConfig
from vit_prisma.sae.train_sae import VisionSAETrainer

def create_sweep(layer):
    sweep_configuration = {
        'method': 'grid',
        'metric': {
            'name': 'overall_loss',
            'goal': 'minimize'
        },
        'parameters': {
            'lr': {
                'values': [3e-4, 1e-3, 3e-3]
            },
            'l1_coefficient': {
                'values': [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
            },
            # 'expansion_factor': {
            #     'values': [16, 32, 64, 128]
            # },
            'hook_point_layer': {
                'values': [layer]  # Set to the specified layer
            }
        },
        'program': 'run_wandb_sweep_sae.py',
    }
    
    # create sweep
    sweep_id = wandb.sweep(sweep_configuration, project=f"tinyclip40M_mlp_out_layer_{layer}_sae_expansion_32")
    print(f"Sweep created with id: {sweep_id} for layer {layer}")

    return sweep_id

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create a sweep for a specific layer')
    parser.add_argument('--layer', type=int, required=True, choices=range(12),
                        help='The layer to create the sweep for (0-11)')
    
    args = parser.parse_args()
    
    sweep_id = create_sweep(args.layer)