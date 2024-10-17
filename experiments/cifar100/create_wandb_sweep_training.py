import wandb

from vit_prisma.utils.wandb_utils import create_sweep

sweep_configuration = {
    'method': 'bayes',
    'metric': {
        'name': 'test_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'lr': {
            'distribution': 'log_uniform',
            'min': 5e-4,
            'max': 5e-3,
        },
        "warmup_steps": {
            "distribution": "uniform",
            "min": 500,
            "max": 3000,
        },
    },
    'program': 'run_wandb_sweep_cifar10.py',
}


if __name__ == '__main__':
    project = "cifar10"
    entity = "Stevinson"

    sweep_config = create_sweep(
        project=project, entity=entity, sweep_cfg=sweep_configuration
    )
