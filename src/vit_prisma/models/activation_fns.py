import numpy as np
import torch
from typing import Dict, Optional, Tuple, Union
from jaxtyping import Float, Int
import torch.nn.functional as F

# activation_dict ={
#     "ReLU": torch.nn.ReLU,
#     "LeakyReLU": torch.nn.LeakyReLU,
#     "gelu_new": gelu_new,
#     "gelu_fast": gelu_fast,
#     "solu": solu,
#     "Linear": torch.nn.Identity,
# }


def gelu_new(
    input: Float[torch.Tensor, "batch pos d_mlp"]
) -> Float[torch.Tensor, "batch pos d_mlp"]:
    # Implementation of GeLU used by GPT2 - subtly different from PyTorch's
    return (
        0.5
        * input
        * (
            1.0
            + torch.tanh(
                np.sqrt(2.0 / np.pi) * (input + 0.044715 * torch.pow(input, 3.0))
            )
        )
    )


def gelu_fast(
    input: Float[torch.Tensor, "batch pos d_mlp"]
) -> Float[torch.Tensor, "batch pos d_mlp"]:
    return (
        0.5
        * input
        * (1.0 + torch.tanh(input * 0.7978845608 * (1.0 + 0.044715 * input * input)))
    )


def solu(
    input: Float[torch.Tensor, "batch pos d_mlp"]
) -> Float[torch.Tensor, "batch pos d_mlp"]:
    """
    SoLU activation function as described by
    https://transformer-circuits.pub/2022/solu/index.html.

    LayerNorm implemented by the MLP class.
    """
    return input * F.softmax(input, dim=-1)