from dataclasses import dataclass
from sae_lens.config import LanguageModelSAERunnerConfig


@dataclass
class VisionModelRunnerConfig(LanguageModelSAERunnerConfig):
    store_batch_size:int = 32 # num images
    
