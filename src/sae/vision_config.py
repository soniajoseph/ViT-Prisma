from dataclasses import dataclass
from sae_lens.training.config import LanguageModelSAERunnerConfig


@dataclass
class VisionModelRunner(LanguageModelSAERunnerConfig):
    total_training_images:int = 1_000_000
