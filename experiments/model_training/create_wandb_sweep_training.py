import wandb


def create_sweep():
    sweep_configuration = {
        'method': 'bayes',
        'metric': {
            'name': 'test_loss',
            'goal': 'minimize'
        },
        'parameters': {
            'lr': {
                'distribution': 'log_uniform',
                'min': 1e-4,
                'max': 1e-2,
            },
        },
        'program': 'imagenet100.py',
    }
    return sweep_configuration


if __name__ == '__main__':
    project = "imagenet100"
    entity = "Stevinson"

    sweep_config = create_sweep()
    sweep_id = wandb.sweep(sweep=sweep_config, project=project, entity=entity)

    print(f"Sweep created with ID: {sweep_id}")
    print(f"To run an agent, use the following command:")
    print(f"wandb agent {entity}/{project}/{sweep_id}")
