import dataclasses

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
