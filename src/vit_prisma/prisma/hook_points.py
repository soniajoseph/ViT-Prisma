"""
For managing the internal activations of a model.

Adapted from TransformerLens, which was inspired by Garcon.
https://github.com/neelnanda-io/TransformerLens/blob/main/transformer_lens/hook_points.py

"""

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch.nn as nn
import torch.utils.hooks as hooks


