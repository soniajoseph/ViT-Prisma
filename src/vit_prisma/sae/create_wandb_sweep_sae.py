import wandb
import torch
import argparse
import math

from vit_prisma.sae.config import VisionModelSAERunnerConfig
from vit_prisma.sae.train_sae import VisionSAETrainer

def create_sweep(layer):
    sweep_configuration = {
        'method': 'bayes', 
        'metric': {
            'name': 'overall_loss',
            'goal': 'minimize'
        },
        'parameters': {
            'lr': {
                'distribution': 'log_uniform',
                'min': math.log(1e-5),  # log(0.00001)
                'max': math.log(1e-1)   # log(0.1)
            },
            'l1_coefficient': {
                'distribution': 'log_uniform',
                'min': math.log(1e-4),  # log(0.0001)
                'max': math.log(1e0)    # log(1.0)
            },
                # 'expansion_factor': {
            #     'values': [32]
            # },
            # 'k': {
            #     'values': [64]
            # },
            'hook_point_layer': {
                'values': [layer]
            },
            'activation_fn_kwargs': {
                'values': [{'k': 32}, {'k': 64}, {'k': 128}]
            },
            'model_name': {
                'values': ['openai/clip-vit-base-patch32']
            }
            
        },
        'program': 'run_wandb_sweep_sae.py',
    }
    return sweep_configuration

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create a sweep for a specific layer')
    parser.add_argument('--layer', type=int, required=True, choices=range(12),
                        help='The layer to create the sweep for (0-11)')
    parser.add_argument('--project', type=str, required=True,)
    parser.add_argument('--entity', type=str, default='perceptual-alignment')
    
    args = parser.parse_args()
    
    sweep_config = create_sweep(args.layer)

    # Initialize the sweep
    sweep_id = wandb.sweep(sweep=sweep_config, project=args.project, entity=args.entity)

    print(f"Sweep created with ID: {sweep_id}")
    print(f"To run an agent, use the following command:")
    print(f"wandb agent {args.entity}/{args.project}/{sweep_id}")