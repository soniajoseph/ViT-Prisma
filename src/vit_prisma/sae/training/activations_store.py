'''
Some code is based off SAELens: https://github.com/jbloomAus/SAELens/blob/main/sae_lens/training/activations_store.py
'''

import contextlib
import json
import os
from typing import Any, Generator, Iterator, Literal, cast

import numpy as np
import torch
from datasets import Dataset, DatasetDict, IterableDataset, load_dataset
from huggingface_hub import hf_hub_download
from huggingface_hub.utils._errors import HfHubHTTPError
from requests import HTTPError
from safetensors import safe_open
from safetensors.torch import save_file
from torch.utils.data import DataLoader
from tqdm import tqdm

from vit_prisma.prisma_tools.hooked_root_module import HookedRootModule

# TO DO!!!