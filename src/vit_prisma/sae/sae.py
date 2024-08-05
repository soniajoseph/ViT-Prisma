'''
Partially based on SAE lens code by Joseph Bloom, and Arthur Conmy's original code
https://github.com/jbloomAus/SAELens/blob/main/sae_lens/sae.py
https://github.com/ArthurConmy/sae/blob/main/sae/model.py

The config here is for inference, not training.

The SAE is also Hooked :) 
'''

import json
import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional, Tuple, TypeVar, Union, overload

import einops
import torch
from jaxtyping import Float
from safetensors.torch import save_file
from torch import nn

from vit_prisma.prisma_tools.hooked_root_module import HookedRootModule, HookPoint
from vit_prisma.sae.config import DTYPE_MAP

from vit_prisma.sae.toolkit.pretrained_sae_loaders import (
    handle_config_defaulting,
    read_sae_from_disk,
)


T = TypeVar("T", bound="SAE")

SPARSITY_PATH = "sparsity.safetensors"
SAE_WEIGHTS_PATH = "sae_weights.safetensors"
SAE_CFG_PATH = "cfg.json"


@dataclass
class SAEConfig:
    # architecture details
    architecture: Literal["standard", "gated", "jumprelu"]

    # forward pass details.
    d_in: int
    d_sae: int
    activation_fn_str: str
    apply_b_dec_to_input: bool
    finetuning_scaling_factor: bool

    # dataset it was trained on details.
    context_size: int
    model_name: str
    hook_name: str
    hook_layer: int
    hook_head_index: Optional[int]
    prepend_bos: bool
    dataset_path: str
    dataset_trust_remote_code: bool
    normalize_activations: str

    # misc
    dtype: str
    device: str
    sae_lens_training_version: Optional[str]
    activation_fn_kwargs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "SAEConfig":

        # rename dict:
        rename_dict = {  # old : new
            "hook_point": "hook_name",
            "hook_point_head_index": "hook_head_index",
            "hook_point_layer": "hook_layer",
            "activation_fn": "activation_fn_str",
        }
        config_dict = {rename_dict.get(k, k): v for k, v in config_dict.items()}

        # use only config terms that are in the dataclass
        config_dict = {
            k: v
            for k, v in config_dict.items()
            if k in cls.__dataclass_fields__  # pylint: disable=no-member
        }
        return cls(**config_dict)

    # def __post_init__(self):

    def to_dict(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture,
            "d_in": self.d_in,
            "d_sae": self.d_sae,
            "dtype": self.dtype,
            "device": self.device,
            "model_name": self.model_name,
            "hook_name": self.hook_name,
            "hook_layer": self.hook_layer,
            "hook_head_index": self.hook_head_index,
            "activation_fn_str": self.activation_fn_str,  # use string for serialization
            "activation_fn_kwargs": self.activation_fn_kwargs or {},
            "apply_b_dec_to_input": self.apply_b_dec_to_input,
            "finetuning_scaling_factor": self.finetuning_scaling_factor,
            "sae_lens_training_version": self.sae_lens_training_version,
            "prepend_bos": self.prepend_bos,
            "dataset_path": self.dataset_path,
            "dataset_trust_remote_code": self.dataset_trust_remote_code,
            "context_size": self.context_size,
            "normalize_activations": self.normalize_activations,
        }


class SAE(HookedRootModule):
    """
    Core Sparse Autoencoder (SAE) class used for inference. For training, see `TrainingSAE`.
    """

    cfg: SAEConfig
    dtype: torch.dtype
    device: torch.device

    # analysis
    use_error_term: bool

    def __init__(
        self,
        cfg: SAEConfig,
        use_error_term: bool = False,
    ):
        super().__init__()

        self.cfg = cfg
        self.activation_fn = get_activation_fn(
            cfg.activation_fn_str, **cfg.activation_fn_kwargs or {}
        )
        self.dtype = DTYPE_MAP[cfg.dtype]
        self.device = torch.device(cfg.device)
        self.use_error_term = use_error_term

        if self.cfg.architecture == "standard":
            self.initialize_weights_basic()
            self.encode = self.encode_standard
        elif self.cfg.architecture == "gated":
            self.initialize_weights_gated()
            self.encode = self.encode_gated
        elif self.cfg.architecture == "jumprelu":
            self.initialize_weights_jumprelu()
            self.encode = self.encode_jumprelu
        else:
            raise (ValueError)

        # handle presence / absence of scaling factor.
        if self.cfg.finetuning_scaling_factor:
            self.apply_finetuning_scaling_factor = (
                lambda x: x * self.finetuning_scaling_factor
            )
        else:
            self.apply_finetuning_scaling_factor = lambda x: x

        # set up hooks
        self.hook_sae_input = HookPoint()
        self.hook_sae_acts_pre = HookPoint()
        self.hook_sae_acts_post = HookPoint()
        self.hook_sae_output = HookPoint()
        self.hook_sae_recons = HookPoint()
        self.hook_sae_error = HookPoint()

        # handle hook_z reshaping if needed.
        # this is very cursed and should be refactored. it exists so that we can reshape out
        # the z activations for hook_z SAEs. but don't know d_head if we split up the forward pass
        # into a separate encode and decode function.
        # this will cause errors if we call decode before encode.
        if self.cfg.hook_name.endswith("_z"):
            self.turn_on_forward_pass_hook_z_reshaping()
        else:
            # need to default the reshape fns
            self.turn_off_forward_pass_hook_z_reshaping()

        # handle run time activation normalization if needed:
        if self.cfg.normalize_activations == "constant_norm_rescale":

            #  we need to scale the norm of the input and store the scaling factor
            def run_time_activation_norm_fn_in(x: torch.Tensor) -> torch.Tensor:
                self.x_norm_coeff = (self.cfg.d_in**0.5) / x.norm(dim=-1, keepdim=True)
                x = x * self.x_norm_coeff
                return x

            def run_time_activation_norm_fn_out(x: torch.Tensor) -> torch.Tensor:  #
                x = x / self.x_norm_coeff
                del self.x_norm_coeff  # prevents reusing
                return x

            self.run_time_activation_norm_fn_in = run_time_activation_norm_fn_in
            self.run_time_activation_norm_fn_out = run_time_activation_norm_fn_out

        elif self.cfg.normalize_activations == "layer_norm":

            #  we need to scale the norm of the input and store the scaling factor
            def run_time_activation_ln_in(
                x: torch.Tensor, eps: float = 1e-5
            ) -> torch.Tensor:
                mu = x.mean(dim=-1, keepdim=True)
                x = x - mu
                std = x.std(dim=-1, keepdim=True)
                x = x / (std + eps)
                self.ln_mu = mu
                self.ln_std = std
                return x

            def run_time_activation_ln_out(x: torch.Tensor, eps: float = 1e-5):
                return x * self.ln_std + self.ln_mu

            self.run_time_activation_norm_fn_in = run_time_activation_ln_in
            self.run_time_activation_norm_fn_out = run_time_activation_ln_out
        else:
            self.run_time_activation_norm_fn_in = lambda x: x
            self.run_time_activation_norm_fn_out = lambda x: x

        self.setup()  # Required for `HookedRootModule`s

    def initialize_weights_basic(self):

        # no config changes encoder bias init for now.
        self.b_enc = nn.Parameter(
            torch.zeros(self.cfg.d_sae, dtype=self.dtype, device=self.device)
        )

        # Start with the default init strategy:
        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(
                    self.cfg.d_sae, self.cfg.d_in, dtype=self.dtype, device=self.device
                )
            )
        )

        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(
                    self.cfg.d_in, self.cfg.d_sae, dtype=self.dtype, device=self.device
                )
            )
        )

        # methdods which change b_dec as a function of the dataset are implemented after init.
        self.b_dec = nn.Parameter(
            torch.zeros(self.cfg.d_in, dtype=self.dtype, device=self.device)
        )

        # scaling factor for fine-tuning (not to be used in initial training)
        # TODO: Make this optional and not included with all SAEs by default (but maintain backwards compatibility)
        if self.cfg.finetuning_scaling_factor:
            self.finetuning_scaling_factor = nn.Parameter(
                torch.ones(self.cfg.d_sae, dtype=self.dtype, device=self.device)
            )

    def initialize_weights_gated(self):
        # Initialize the weights and biases for the gated encoder
        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(
                    self.cfg.d_in, self.cfg.d_sae, dtype=self.dtype, device=self.device
                )
            )
        )

        self.b_gate = nn.Parameter(
            torch.zeros(self.cfg.d_sae, dtype=self.dtype, device=self.device)
        )

        self.r_mag = nn.Parameter(
            torch.zeros(self.cfg.d_sae, dtype=self.dtype, device=self.device)
        )

        self.b_mag = nn.Parameter(
            torch.zeros(self.cfg.d_sae, dtype=self.dtype, device=self.device)
        )

        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(
                    self.cfg.d_sae, self.cfg.d_in, dtype=self.dtype, device=self.device
                )
            )
        )

        self.b_dec = nn.Parameter(
            torch.zeros(self.cfg.d_in, dtype=self.dtype, device=self.device)
        )

    def initialize_weights_jumprelu(self):
        # The params are identical to the standard SAE
        # except we use a threshold parameter too
        self.threshold = nn.Parameter(
            torch.zeros(self.cfg.d_sae, dtype=self.dtype, device=self.device)
        )
        self.b_enc = nn.Parameter(
            torch.zeros(self.cfg.d_sae, dtype=self.dtype, device=self.device)
        )

        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(
                    self.cfg.d_sae, self.cfg.d_in, dtype=self.dtype, device=self.device
                )
            )
        )
        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(
                    self.cfg.d_in, self.cfg.d_sae, dtype=self.dtype, device=self.device
                )
            )
        )
        self.b_dec = nn.Parameter(
            torch.zeros(self.cfg.d_in, dtype=self.dtype, device=self.device)
        )

    @overload
    def to(
        self: T,
        device: Optional[Union[torch.device, str]] = ...,
        dtype: Optional[torch.dtype] = ...,
        non_blocking: bool = ...,
    ) -> T: ...

    @overload
    def to(self: T, dtype: torch.dtype, non_blocking: bool = ...) -> T: ...

    @overload
    def to(self: T, tensor: torch.Tensor, non_blocking: bool = ...) -> T: ...

    def to(self, *args: Any, **kwargs: Any) -> "SAE":  # type: ignore
        device_arg = None
        dtype_arg = None

        # Check args
        for arg in args:
            if isinstance(arg, (torch.device, str)):
                device_arg = arg
            elif isinstance(arg, torch.dtype):
                dtype_arg = arg
            elif isinstance(arg, torch.Tensor):
                device_arg = arg.device
                dtype_arg = arg.dtype

        # Check kwargs
        device_arg = kwargs.get("device", device_arg)
        dtype_arg = kwargs.get("dtype", dtype_arg)

        if device_arg is not None:
            # Convert device to torch.device if it's a string
            device = (
                torch.device(device_arg) if isinstance(device_arg, str) else device_arg
            )

            # Update the cfg.device
            self.cfg.device = str(device)

            # Update the .device property
            self.device = device

        if dtype_arg is not None:
            # Update the cfg.dtype
            self.cfg.dtype = str(dtype_arg)

            # Update the .dtype property
            self.dtype = dtype_arg

        # Call the parent class's to() method to handle all cases (device, dtype, tensor)
        return super().to(*args, **kwargs)

    # Basic Forward Pass Functionality.
    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        feature_acts = self.encode(x)
        sae_out = self.decode(feature_acts)

        # TEMP
        if self.use_error_term and self.cfg.architecture == "standard":
            with torch.no_grad():
                # Recompute everything without hooks to get true error term
                # Otherwise, the output with error term will always equal input, even for causal interventions that affect x_reconstruct
                # This is in a no_grad context to detach the error, so we can compute SAE feature gradients (eg for attribution patching). See A.3 in https://arxiv.org/pdf/2403.19647.pdf for more detail
                # NOTE: we can't just use `sae_error = input - x_reconstruct.detach()` or something simpler, since this would mean intervening on features would mean ablating features still results in perfect reconstruction.

                # move x to correct dtype
                x = x.to(self.dtype)

                # handle hook z reshaping if needed.
                sae_in = self.reshape_fn_in(x)  # type: ignore

                # handle run time activation normalization if needed
                sae_in = self.run_time_activation_norm_fn_in(sae_in)

                # apply b_dec_to_input if using that method.
                sae_in_cent = sae_in - (self.b_dec * self.cfg.apply_b_dec_to_input)

                # "... d_in, d_in d_sae -> ... d_sae",
                hidden_pre = sae_in_cent @ self.W_enc + self.b_enc
                feature_acts = self.activation_fn(hidden_pre)
                x_reconstruct_clean = self.reshape_fn_out(
                    self.apply_finetuning_scaling_factor(feature_acts) @ self.W_dec
                    + self.b_dec,
                    d_head=self.d_head,
                )

                sae_out = self.run_time_activation_norm_fn_out(sae_out)
                sae_error = self.hook_sae_error(x - x_reconstruct_clean)
            return self.hook_sae_output(sae_out + sae_error)

        # TODO: Add tests
        elif self.use_error_term and self.cfg.architecture == "gated":
            with torch.no_grad():
                x = x.to(self.dtype)
                sae_in = self.reshape_fn_in(x)  # type: ignore
                gating_pre_activation = sae_in @ self.W_enc + self.b_gate
                active_features = (gating_pre_activation > 0).float()

                # Magnitude path with weight sharing
                magnitude_pre_activation = self.hook_sae_acts_pre(
                    sae_in @ (self.W_enc * self.r_mag.exp()) + self.b_mag
                )
                feature_magnitudes = self.hook_sae_acts_post(
                    self.activation_fn(magnitude_pre_activation)
                )
                feature_acts_clean = active_features * feature_magnitudes
                x_reconstruct_clean = self.reshape_fn_out(
                    self.apply_finetuning_scaling_factor(feature_acts_clean)
                    @ self.W_dec
                    + self.b_dec,
                    d_head=self.d_head,
                )

                sae_error = self.hook_sae_error(x - x_reconstruct_clean)
            return self.hook_sae_output(sae_out + sae_error)

        # TODO: Add tests
        elif self.use_error_term and self.cfg.architecture == "jumprelu":
            with torch.no_grad():
                x = x.to(self.dtype)
                sae_in = self.reshape_fn_in(x)  # type: ignore

                # handle run time activation normalization if needed
                x = self.run_time_activation_norm_fn_in(x)

                # apply b_dec_to_input if using that method.
                sae_in = x - (self.b_dec * self.cfg.apply_b_dec_to_input)

                # "... d_in, d_in d_sae -> ... d_sae",
                hidden_pre = sae_in @ self.W_enc + self.b_enc
                feature_acts = self.hook_sae_acts_post(
                    self.activation_fn(hidden_pre) * (hidden_pre > self.threshold)
                )
                x_reconstruct_clean = self.reshape_fn_out(
                    self.apply_finetuning_scaling_factor(feature_acts) @ self.W_dec
                    + self.b_dec,
                    d_head=self.d_head,  # TODO(conmy): d_head?! Eh?
                )
                sae_error = self.hook_sae_error(x - x_reconstruct_clean)
            return self.hook_sae_output(sae_out + sae_error)
        elif self.use_error_term:
            raise ValueError(f"No error term implemented for {self.cfg.architecture=}")

        return self.hook_sae_output(sae_out)

    def encode_gated(
        self, x: Float[torch.Tensor, "... d_in"]
    ) -> Float[torch.Tensor, "... d_sae"]:

        x = x.to(self.dtype)
        x = self.reshape_fn_in(x)
        x = self.hook_sae_input(x)
        x = self.run_time_activation_norm_fn_in(x)
        sae_in = x - self.b_dec * self.cfg.apply_b_dec_to_input

        # Gating path
        gating_pre_activation = sae_in @ self.W_enc + self.b_gate
        active_features = (gating_pre_activation > 0).to(self.dtype)

        # Magnitude path with weight sharing
        magnitude_pre_activation = self.hook_sae_acts_pre(
            sae_in @ (self.W_enc * self.r_mag.exp()) + self.b_mag
        )
        feature_magnitudes = self.hook_sae_acts_post(
            self.activation_fn(magnitude_pre_activation)
        )

        return active_features * feature_magnitudes

    def encode_jumprelu(
        self, x: Float[torch.Tensor, "... d_in"]
    ) -> Float[torch.Tensor, "... d_sae"]:
        """
        Calculate SAE features from inputs
        """

        # move x to correct dtype
        x = x.to(self.dtype)

        # handle hook z reshaping if needed.
        x = self.reshape_fn_in(x)  # type: ignore

        # handle run time activation normalization if needed
        x = self.run_time_activation_norm_fn_in(x)

        # apply b_dec_to_input if using that method.
        sae_in = self.hook_sae_input(x - (self.b_dec * self.cfg.apply_b_dec_to_input))

        # "... d_in, d_in d_sae -> ... d_sae",
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)

        feature_acts = self.hook_sae_acts_post(
            self.activation_fn(hidden_pre) * (hidden_pre > self.threshold)
        )

        return feature_acts

    def encode_standard(
        self, x: Float[torch.Tensor, "... d_in"]
    ) -> Float[torch.Tensor, "... d_sae"]:
        """
        Calculate SAE features from inputs
        """

        x = x.to(self.dtype)
        x = self.reshape_fn_in(x)
        x = self.hook_sae_input(x)
        x = self.run_time_activation_norm_fn_in(x)

        # apply b_dec_to_input if using that method.
        sae_in = x - (self.b_dec * self.cfg.apply_b_dec_to_input)

        # "... d_in, d_in d_sae -> ... d_sae",
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)
        feature_acts = self.hook_sae_acts_post(self.activation_fn(hidden_pre))

        return feature_acts

    def decode(
        self, feature_acts: Float[torch.Tensor, "... d_sae"]
    ) -> Float[torch.Tensor, "... d_in"]:
        """Decodes SAE feature activation tensor into a reconstructed input activation tensor."""
        # "... d_sae, d_sae d_in -> ... d_in",
        sae_out = self.hook_sae_recons(
            self.apply_finetuning_scaling_factor(feature_acts) @ self.W_dec + self.b_dec
        )

        # handle run time activation normalization if needed
        # will fail if you call this twice without calling encode in between.
        sae_out = self.run_time_activation_norm_fn_out(sae_out)

        # handle hook z reshaping if needed.
        sae_out = self.reshape_fn_out(sae_out, self.d_head)  # type: ignore

        return sae_out

    @torch.no_grad()
    def fold_W_dec_norm(self):
        W_dec_norms = self.W_dec.norm(dim=-1).unsqueeze(1)
        self.W_dec.data = self.W_dec.data / W_dec_norms
        self.W_enc.data = self.W_enc.data * W_dec_norms.T
        if self.cfg.architecture == "gated":
            self.r_mag.data = self.r_mag.data * W_dec_norms.squeeze()
            self.b_gate.data = self.b_gate.data * W_dec_norms.squeeze()
            self.b_mag.data = self.b_mag.data * W_dec_norms.squeeze()
        else:
            self.b_enc.data = self.b_enc.data * W_dec_norms.squeeze()

    @torch.no_grad()
    def fold_activation_norm_scaling_factor(
        self, activation_norm_scaling_factor: float
    ):
        self.W_enc.data = self.W_enc.data * activation_norm_scaling_factor
        # previously weren't doing this.
        self.W_dec.data = self.W_dec.data / activation_norm_scaling_factor

        # once we normalize, we shouldn't need to scale activations.
        self.cfg.normalize_activations = "none"

    def save_model(self, path: str, sparsity: Optional[torch.Tensor] = None):

        if not os.path.exists(path):
            os.mkdir(path)

        # generate the weights
        save_file(self.state_dict(), f"{path}/{SAE_WEIGHTS_PATH}")

        # save the config
        config = self.cfg.to_dict()

        with open(f"{path}/{SAE_CFG_PATH}", "w") as f:
            json.dump(config, f)

        if sparsity is not None:
            sparsity_in_dict = {"sparsity": sparsity}
            save_file(sparsity_in_dict, f"{path}/{SPARSITY_PATH}")  # type: ignore

    @classmethod
    def load_from_pretrained(
        cls, path: str, device: str = "cpu", dtype: str | None = None
    ) -> "SAE":

        # get the config
        config_path = os.path.join(path, SAE_CFG_PATH)
        with open(config_path, "r") as f:
            cfg_dict = json.load(f)
        cfg_dict = handle_config_defaulting(cfg_dict)
        cfg_dict["device"] = device
        if dtype is not None:
            cfg_dict["dtype"] = dtype

        weight_path = os.path.join(path, SAE_WEIGHTS_PATH)
        cfg_dict, state_dict = read_sae_from_disk(
            cfg_dict=cfg_dict,
            weight_path=weight_path,
            device=device,
            dtype=DTYPE_MAP[cfg_dict["dtype"]],
        ) 

        sae_cfg = SAEConfig.from_dict(cfg_dict)

        sae = cls(sae_cfg)
        sae.load_state_dict(state_dict)

        return sae

    @classmethod
    def from_pretrained(
        cls,
        release: str,
        sae_id: str,
        device: str = "cpu",
    ) -> Tuple["SAE", dict[str, Any], Optional[torch.Tensor]]:
        """

        Load a pretrained SAE from the Hugging Face model hub.

        Args:
            release: The release name. This will be mapped to a huggingface repo id based on the pretrained_saes.yaml file.
            id: The id of the SAE to load. This will be mapped to a path in the huggingface repo.
            device: The device to load the SAE on.
            return_sparsity_if_present: If True, will return the log sparsity tensor if it is present in the model directory in the Hugging Face model hub.
        """

        # Raise not implemented yet and return
        raise NotImplementedError("This method is not implemented yet.")

        # get sae directory
        
        # sae_directory = get_pretrained_saes_directory()

        # # get the repo id and path to the SAE
        # if release not in sae_directory:
        #     raise ValueError(
        #         f"Release {release} not found in pretrained SAEs directory."
        #     )
        # if sae_id not in sae_directory[release].saes_map:
        #     raise ValueError(
        #         f"ID {sae_id} not found in release {release}. Valid IDs are {sae_directory[release].saes_map.keys()}"
        #     )
        # sae_info = sae_directory[release]
        # hf_repo_id = sae_info.repo_id
        # hf_path = sae_info.saes_map[sae_id]

        # conversion_loader_name = sae_info.conversion_func or "sae_lens"
        # if conversion_loader_name not in NAMED_PRETRAINED_SAE_LOADERS:
        #     raise ValueError(
        #         f"Conversion func {conversion_loader_name} not found in NAMED_PRETRAINED_SAE_LOADERS."
        #     )
        # conversion_loader = NAMED_PRETRAINED_SAE_LOADERS[conversion_loader_name]

        # cfg_dict, state_dict, log_sparsities = conversion_loader(
        #     repo_id=hf_repo_id,
        #     folder_name=hf_path,
        #     device=device,
        #     force_download=False,
        #     cfg_overrides=sae_directory[release].config_overrides,
        # )

        # sae = cls(SAEConfig.from_dict(cfg_dict))
        # sae.load_state_dict(state_dict)

        # # Check if normalization is 'expected_average_only_in'
        # if cfg_dict.get("normalize_activations") == "expected_average_only_in":
        #     norm_scaling_factor = get_norm_scaling_factor(release, sae_id)
        #     if norm_scaling_factor is not None:
        #         sae.fold_activation_norm_scaling_factor(norm_scaling_factor)
        #         cfg_dict["normalize_activations"] = "none"
        #     else:
        #         warnings.warn(
        #             f"norm_scaling_factor not found for {release} and {sae_id}, but normalize_activations is 'expected_average_only_in'. Skipping normalization folding."
        #         )

        # return sae, cfg_dict, log_sparsities

    def get_name(self):
        sae_name = f"sae_{self.cfg.model_name}_{self.cfg.hook_name}_{self.cfg.d_sae}"
        return sae_name

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "SAE":
        return cls(SAEConfig.from_dict(config_dict))

    def turn_on_forward_pass_hook_z_reshaping(self):

        assert self.cfg.hook_name.endswith(
            "_z"
        ), "This method should only be called for hook_z SAEs."

        def reshape_fn_in(x: torch.Tensor):
            self.d_head = x.shape[-1]  # type: ignore
            self.reshape_fn_in = lambda x: einops.rearrange(
                x, "... n_heads d_head -> ... (n_heads d_head)"
            )
            return einops.rearrange(x, "... n_heads d_head -> ... (n_heads d_head)")

        self.reshape_fn_in = reshape_fn_in

        self.reshape_fn_out = lambda x, d_head: einops.rearrange(
            x, "... (n_heads d_head) -> ... n_heads d_head", d_head=d_head
        )
        self.hook_z_reshaping_mode = True

    def turn_off_forward_pass_hook_z_reshaping(self):
        self.reshape_fn_in = lambda x: x
        self.reshape_fn_out = lambda x, d_head: x
        self.d_head = None
        self.hook_z_reshaping_mode = False


class TopK(nn.Module):
    def __init__(
        self, k: int, postact_fn: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU()
    ):
        super().__init__()
        self.k = k
        self.postact_fn = postact_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        topk = torch.topk(x, k=self.k, dim=-1)
        values = self.postact_fn(topk.values)
        result = torch.zeros_like(x)
        result.scatter_(-1, topk.indices, values)
        return result


def get_activation_fn(
    activation_fn: str, **kwargs: Any
) -> Callable[[torch.Tensor], torch.Tensor]:
    if activation_fn == "relu":
        return torch.nn.ReLU()
    elif activation_fn == "tanh-relu":

        def tanh_relu(input: torch.Tensor) -> torch.Tensor:
            input = torch.relu(input)
            input = torch.tanh(input)
            return input

        return tanh_relu
    elif activation_fn == "topk":
        assert "k" in kwargs, "TopK activation function requires a k value."
        k = kwargs.get("k", 1)  # Default k to 1 if not provided
        postact_fn = kwargs.get(
            "postact_fn", nn.ReLU()
        )  # Default post-activation to ReLU if not provided

        return TopK(k, postact_fn)
    else:
        raise ValueError(f"Unknown activation function: {activation_fn}")

