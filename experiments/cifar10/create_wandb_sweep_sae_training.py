import wandb

from vit_prisma.utils.wandb_utils import create_sweep

sweep_configuration = {
    'method': 'bayes',
    'metric': {
        'name': 'metrics/l0',
        'goal': 'minimize'
    },
    'early_terminate': {
        'type': 'hyperband',
        'min_iter': 2_000_000,
    },
    'parameters': {
        'lr': {
            'distribution': 'log_uniform_values',
            'min': 0.005,
            'max': 0.02,
        },
        "l1_coefficient": {
            "distribution": "log_uniform_values",
            "min": 0.5,
            "max": 1,
        },
    },
    'run_cap': 10,
    'program': 'experiments/cifar10/run_wandb_sweep_train_sae_cifar10.py',
}


if __name__ == '__main__':
    project = "cifar10"
    entity = "Stevinson"

    sweep_config = create_sweep(
        project=project, entity=entity, sweep_cfg=sweep_configuration,
    )
