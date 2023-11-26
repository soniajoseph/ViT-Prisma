from enum import Enum

class Objective(Enum):
    CLASSIFICATION = 1
    GENERATION = 2

class Masking(Enum):
    NONE = 1
    RANDOM = 2
    AUTOREGRESSIVE = 3