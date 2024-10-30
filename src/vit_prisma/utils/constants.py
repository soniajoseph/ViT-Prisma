from enum import Enum


class Evaluation(Enum):
    """Enum that maps to the possible evaluation functions"""

    FEATURE_BASIS_EVAL = "evaluate_sae_features"
    NEURON_BASIS_EVAL = "find_top_activations_for_neurons"


class EvaluationContext(Enum):

    TRAINING = "train"
    POST_TRAINING = "post-train"