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