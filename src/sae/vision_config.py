from dataclasses import dataclass
from sae_lens.training.config import LanguageModelSAERunnerConfig
import yaml

@dataclass
class VisionModelRunnerConfig(LanguageModelSAERunnerConfig):
    store_batch_size:int =32 # num images

    @classmethod
    def from_yaml(cls, path:str):
        with open(path, 'r') as file:
            config_dict = yaml.safe_load(file)

        config_dict['arguments']['hook_point_layer'] = config_dict['arguments']['layers']
        del config_dict['arguments']['layers']

        return cls(**config_dict['arguments'])
