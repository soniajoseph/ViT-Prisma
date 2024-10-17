import dataclasses

import wandb


def update_dataclass_from_dict(dc, dct):
    for key, value in dct.items():
        if hasattr(dc, key):
            attr = getattr(dc, key)
            if dataclasses.is_dataclass(attr):
                update_dataclass_from_dict(attr, value)
            else:
                setattr(dc, key, value)

def update_config_with_wandb_sweep(config, sweep_values):
    update_dataclass_from_dict(config, sweep_values)

def dataclass_to_dict(obj):
    if dataclasses.is_dataclass(obj):
        return {name: dataclass_to_dict(value) for name, value in dataclasses.asdict(obj).items()}
    else:
        return obj


def create_sweep(project: str, sweep_cfg: dict, entity: str = None):
    sweep_id = wandb.sweep(sweep=sweep_cfg, project=project, entity=entity)

    print(f"Sweep created with ID: {sweep_id}")
    print(f"To run an agent, use the following command:")
    print(f"wandb agent {entity}/{project}/{sweep_id}")
