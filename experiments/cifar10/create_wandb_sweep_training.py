import wandb

from vit_prisma.utils.wandb_utils import create_sweep

sweep_configuration = {
    'method': 'bayes',
    'metric': {
        'name': 'test_loss',
        'goal': 'minimize'
    },
    'early_terminate': {
        'type': 'hyperband',
        'min_iter': 2_000_000,
    },
    'parameters': {
        'lr': {
            'distribution': 'log_uniform_values',
            'min': 4e-4,
            'max': 6e-4,
        },
        "warmup_steps": {
            "distribution": "int_uniform",
            "min": 900,
            "max": 1100,
        },
    },
    'command': {
        'count': 1,
    },
    'run_cap': 1,
    'program': 'experiments/cifar10/run_wandb_sweep_cifar10.py',
}


if __name__ == '__main__':
    project = "cifar10"
    entity = "Stevinson"

    sweep_config = create_sweep(
        project=project, entity=entity, sweep_cfg=sweep_configuration,
    )
