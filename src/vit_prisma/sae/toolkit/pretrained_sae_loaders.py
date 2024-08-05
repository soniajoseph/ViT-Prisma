'''
Based on Joseph Bloom repo: https://github.com/jbloomAus/SAELens/blob/main/sae_lens/toolkit/pretrained_sae_loaders.py
'''

import json
import re
from typing import Any, Dict, Optional, Protocol

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.utils._errors import EntryNotFoundError
from safetensors import safe_open



# loaders take in a repo_id, folder_name, device, and whether to force download, and returns a tuple of config and state_dict
class PretrainedSaeLoader(Protocol):

    def __call__(
        self,
        repo_id: str,
        folder_name: str,
        device: str | torch.device | None = None,
        force_download: bool = False,
        cfg_overrides: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, torch.Tensor], Optional[torch.Tensor]]: ...


def sae_lens_loader(
    repo_id: str,
    folder_name: str,
    device: str = "cpu",
    force_download: bool = False,
    cfg_overrides: Optional[dict[str, Any]] = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], Optional[torch.Tensor]]:
    """
    Get's SAEs from HF, loads them.
    """
    # Get the config
    cfg_dict = get_sae_config_from_hf(
        repo_id,
        folder_name,
        force_download,
    )
    # Apply overrides if provided
    if cfg_overrides is not None:
        cfg_dict.update(cfg_overrides)
    cfg_dict["device"] = device
    cfg_dict = handle_config_defaulting(cfg_dict)

    weights_filename = f"{folder_name}/sae_weights.safetensors"
    sae_path = hf_hub_download(
        repo_id=repo_id, filename=weights_filename, force_download=force_download
    )

    # TODO: Make this cleaner. I hate try except statements.
    try:
        sparsity_filename = f"{folder_name}/sparsity.safetensors"
        log_sparsity_path = hf_hub_download(
            repo_id=repo_id, filename=sparsity_filename, force_download=force_download
        )
    except EntryNotFoundError:
        log_sparsity_path = None  # no sparsity file

    cfg_dict, state_dict = read_sae_from_disk(
        cfg_dict=cfg_dict,
        weight_path=sae_path,
        device=device,
    )

    # get sparsity tensor if it exists
    if log_sparsity_path is not None:
        with safe_open(log_sparsity_path, framework="pt", device=device) as f:  # type: ignore
            log_sparsity = f.get_tensor("sparsity")
    else:
        log_sparsity = None

    return cfg_dict, state_dict, log_sparsity

def read_sae_from_disk(
    cfg_dict: dict[str, Any],
    weight_path: str,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """
    Given a loaded dictionary and a path to a weight file, load the weights and return the state_dict.
    """

    state_dict = {}
    with safe_open(weight_path, framework="pt", device=device) as f:  # type: ignore
        for k in f.keys():
            state_dict[k] = f.get_tensor(k).to(dtype=dtype)

    # if bool and True, then it's the April update method of normalizing activations and hasn't been folded in.
    if "scaling_factor" in state_dict:
        # we were adding it anyway for a period of time but are no longer doing so.
        # so we should delete it if
        if torch.allclose(
            state_dict["scaling_factor"],
            torch.ones_like(state_dict["scaling_factor"]),
        ):
            del state_dict["scaling_factor"]
            cfg_dict["finetuning_scaling_factor"] = False
        else:
            assert cfg_dict[
                "finetuning_scaling_factor"
            ], "Scaling factor is present but finetuning_scaling_factor is False."
            state_dict["finetuning_scaling_factor"] = state_dict["scaling_factor"]
            del state_dict["scaling_factor"]
    else:
        # it's there and it's not all 1's, we should use it.
        cfg_dict["finetuning_scaling_factor"] = False

    return cfg_dict, state_dict


def handle_config_defaulting(cfg_dict: dict[str, Any]) -> dict[str, Any]:

    # Set default values for backwards compatibility
    cfg_dict.setdefault("prepend_bos", True)
    cfg_dict.setdefault("dataset_trust_remote_code", True)
    cfg_dict.setdefault("apply_b_dec_to_input", True)
    cfg_dict.setdefault("finetuning_scaling_factor", False)
    cfg_dict.setdefault("sae_lens_training_version", None)
    cfg_dict.setdefault("activation_fn_str", cfg_dict.get("activation_fn", "relu"))
    cfg_dict.setdefault("architecture", "standard")

    if "normalize_activations" in cfg_dict and isinstance(
        cfg_dict["normalize_activations"], bool
    ):
        # backwards compatibility
        cfg_dict["normalize_activations"] = (
            "none"
            if not cfg_dict["normalize_activations"]
            else "expected_average_only_in"
        )

    cfg_dict.setdefault("normalize_activations", "none")

    return cfg_dict

def get_sae_config_from_hf(
    repo_id: str,
    folder_name: str,
    force_download: bool = False,
) -> Dict[str, Any]:
    """
    Retrieve the configuration for a Sparse Autoencoder (SAE) from a Hugging Face repository.

    Args:
        repo_id (str): The repository ID on Hugging Face.
        folder_name (str): The folder name within the repository containing the config file.
        force_download (bool, optional): Whether to force download the config file. Defaults to False.
        cfg_overrides (Optional[Dict[str, Any]], optional): Overrides for the config. Defaults to None.

    Returns:
        Dict[str, Any]: The configuration dictionary for the SAE.
    """
    cfg_filename = f"{folder_name}/cfg.json"
    cfg_path = hf_hub_download(
        repo_id=repo_id, filename=cfg_filename, force_download=force_download
    )

    with open(cfg_path, "r") as f:
        cfg_dict = json.load(f)

    return cfg_dict