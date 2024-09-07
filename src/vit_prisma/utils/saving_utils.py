import json
import torch

from pathlib import Path
import re


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


# Get current Prisma version
def get_version():
    # Read the version from setup.py
    setup_py_path = Path(__file__).parent.parent.parent.parent 
    setup_py_path = setup_py_path / 'setup.py'

    # Check if setup.py exists
    if not setup_py_path.exists():
        raise FileNotFoundError(f"setup.py not found at {setup_py_path}")
    
    # Read the content of setup.py
    with open(setup_py_path, 'r') as f:
        content = f.read()

    # Try to find version using regex
    version_match = re.search(r"version\s*=\s*['\"]([^'\"]+)['\"]", content)
    if version_match:
        return version_match.group(1)

    # If version is not found, print the content of setup.py
    print("Content of setup.py:")
    print(content)
    
    raise ValueError("Version not found in setup.py. Content of the file is printed above.")