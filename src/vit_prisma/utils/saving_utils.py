import json
import torch
# Save config along with trainig files


def object_to_dict(obj):
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    elif isinstance(obj, dict):
        return {k: object_to_dict(v) for k, v in obj.items() if not callable(v)}
    elif isinstance(obj, list):
        return [object_to_dict(item) for item in obj]
    elif hasattr(obj, "__dict__"):  # for custom objects
        # Filter out methods or built-in functions
        return object_to_dict({k: v for k, v in obj.__dict__.items() if not callable(v)})
    elif isinstance(obj, torch.dtype):
        return str(obj)
    else:
        raise TypeError(f"Object of type {type(obj)} is not serializable to JSON")

# Function to save config to a JSON file
def save_config_to_file(config, file_path):
    with open(file_path, 'w') as f:
        json.dump(object_to_dict(config), f, indent=4)

