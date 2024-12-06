"""Parent class for both the hooked text and vision encoder models."""
import logging
from typing import Optional, Literal
from typing import Union, Dict

import torch
from fancy_einsum import einsum
from transformers import ViTForImageClassification

from vit_prisma.prisma_tools import FactoredMatrix
from vit_prisma.prisma_tools.hooked_root_module import HookedRootModule
from vit_prisma.prisma_tools.loading_from_pretrained import convert_pretrained_model_config, get_pretrained_state_dict
from vit_prisma.prisma_tools.loading_from_pretrained import fill_missing_keys
from vit_prisma.utils.enums import ModelType
from vit_prisma.utils.prisma_utils import transpose

DTYPE_FROM_STRING = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


class HookedTransformer(HookedRootModule):

    def __init__(self):
        super().__init__()


    def load_and_process_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        fold_ln: Optional[bool] = True,
        center_writing_weights: Optional[bool] = True,
        fold_value_biases: Optional[bool] = True,
        refactor_factored_attn_matrices: Optional[bool] = False,
    ):
        """Load & Process State Dict.

        Load a state dict into the model, and to apply processing to simplify it. The state dict is
        assumed to be in the HookedTransformer format.

        See the relevant method (same name as the flag) for more details on the folding, centering
        and processing flags.

        Args:
            state_dict (dict): The state dict of the model, in HookedTransformer format. fold_ln
            fold_ln (bool, optional): Whether to fold in the LayerNorm weights to the
                subsequent linear layer. This does not change the computation. Defaults to True.
            center_writing_weights (bool, optional): Whether to center weights writing to the
                residual stream (ie set mean to be zero). Due to LayerNorm this doesn't change the
                computation. Defaults to True.
            fold_value_biases (bool, optional): Whether to fold the value biases into the output
                bias. Because attention patterns add up to 1, the value biases always have a
                constant effect on a layer's output, and it doesn't matter which head a bias is
                associated with. We can factor this all into a single output bias to the layer, and
                make it easier to interpret the head's output.
            refactor_factored_attn_matrices (bool, optional): Whether to convert the factored
                matrices (W_Q & W_K, and W_O & W_V) to be "even". Defaults to False.
            model_name (str, optional): checks the model name for special cases of state dict
                loading. Only used for Redwood 2L model currently.
        """
        if self.cfg.dtype not in [torch.float32, torch.float64] and fold_ln:
            logging.warning(
                "With reduced precision, it is advised to use `from_pretrained_no_processing` instead of `from_pretrained`."
            )

        state_dict = fill_missing_keys(self, state_dict)
        if fold_ln:
            if self.cfg.normalization_type in ["LN", "LNPre"]:
                state_dict = self.fold_layer_norm(state_dict)
            elif self.cfg.normalization_type in ["RMS", "RMSPre"]:
                state_dict = self.fold_layer_norm(
                    state_dict, fold_biases=False, center_weights=False
                )
            else:
                logging.warning(
                    "You are not using LayerNorm or RMSNorm, so the layer norm weights can't be folded! Skipping"
                )

        if center_writing_weights:
            if self.cfg.normalization_type not in ["LN", "LNPre"]:
                logging.warning(
                    "You are not using LayerNorm, so the writing weights can't be centered! Skipping"
                )
            elif self.cfg.final_rms:
                logging.warning(
                    "This model is using final RMS normalization, so the writing weights can't be centered! Skipping"
                )
            else:
                state_dict = self.center_writing_weights(state_dict)

        if fold_value_biases:
            state_dict = self.fold_value_biases(state_dict)

        if refactor_factored_attn_matrices:
            state_dict = self.refactor_factored_attn_matrices(state_dict)

        self.load_state_dict(state_dict, strict=True)

    def center_writing_weights(self, state_dict: Dict[str, torch.Tensor]):
        """Center Writing Weights.

        Centers the weights of the model that write to the residual stream - W_out, W_E, W_pos and
        W_out. This is done by subtracting the mean of the weights from the weights themselves. This
        is done in-place. See fold_layer_norm for more details.
        """
        # state_dict["embed.W_E"] = state_dict["embed.W_E"] - state_dict[
        #     "embed.W_E"
        # ].mean(-1, keepdim=True)
        if self.cfg.positional_embedding_type != "rotary":
            state_dict["pos_embed.W_pos"] = state_dict["pos_embed.W_pos"] - state_dict[
                "pos_embed.W_pos"
            ].mean(-1, keepdim=True)
        for l in range(self.cfg.n_layers):
            state_dict[f"blocks.{l}.attn.W_O"] = state_dict[
                f"blocks.{l}.attn.W_O"
            ] - state_dict[f"blocks.{l}.attn.W_O"].mean(
                -1, keepdim=True
            )  # W_O is [head_index, d_model, d_head]
            state_dict[f"blocks.{l}.attn.b_O"] = (
                state_dict[f"blocks.{l}.attn.b_O"]
                - state_dict[f"blocks.{l}.attn.b_O"].mean()
            )  # b_O is [d_model]
            if not self.cfg.attn_only:
                state_dict[f"blocks.{l}.mlp.W_out"] = state_dict[
                    f"blocks.{l}.mlp.W_out"
                ] - state_dict[f"blocks.{l}.mlp.W_out"].mean(-1, keepdim=True)
                state_dict[f"blocks.{l}.mlp.b_out"] = (
                    state_dict[f"blocks.{l}.mlp.b_out"]
                    - state_dict[f"blocks.{l}.mlp.b_out"].mean()
                )

        print("Centered weights writing to residual stream")
        return state_dict

    def fold_value_biases(self, state_dict: Dict[str, torch.Tensor]):
        """Fold the value biases into the output bias.

        Because attention patterns add up to 1, the value biases always have a constant effect on a
        head's output. Further, as the outputs of each head in a layer add together, each head's
        value bias has a constant effect on the *layer's* output, which can make it harder to
        interpret the effect of any given head, and it doesn't matter which head a bias is
        associated with. We can factor this all into a single output bias to the layer, and make it
        easier to interpret the head's output. Formally, we take b_O_new = b_O_original +
        sum_head(b_V_head @ W_O_head).
        """
        for layer in range(self.cfg.n_layers):
            # shape [head_index, d_head]
            if self.cfg.n_key_value_heads is None:
                b_V = state_dict[f"blocks.{layer}.attn.b_V"]
            else:
                b_V = state_dict[f"blocks.{layer}.attn._b_V"]
                b_V = torch.repeat_interleave(
                    b_V, dim=0, repeats=self.cfg.n_heads // self.cfg.n_key_value_heads
                )
            # [head_index, d_head, d_model]
            W_O = state_dict[f"blocks.{layer}.attn.W_O"]
            # [d_model]
            b_O_original = state_dict[f"blocks.{layer}.attn.b_O"]
            folded_b_O = b_O_original + (b_V[:, :, None] * W_O).sum([0, 1])

            state_dict[f"blocks.{layer}.attn.b_O"] = folded_b_O
            if self.cfg.n_key_value_heads is None:
                state_dict[f"blocks.{layer}.attn.b_V"] = torch.zeros_like(b_V)
            else:
                state_dict[f"blocks.{layer}.attn._b_V"] = torch.zeros_like(
                    state_dict[f"blocks.{layer}.attn._b_V"]
                )

        return state_dict

    def refactor_factored_attn_matrices(self, state_dict: Dict[str, torch.Tensor]):
        """Experimental method for managing queries, keys and values.

        As argued in [A Mathematical Framework for Transformer
        Circuits](https://transformer-circuits.pub/2021/framework/index.html), queries, keys and
        values are somewhat arbitrary intermediate terms when computing with the low rank factored
        matrices W_QK = W_Q @ W_K.T and W_OV = W_V @ W_O, and these matrices are the only thing
        determining head behaviour. But there are many ways to find a low rank factorization to a
        given matrix, and hopefully some of these are more interpretable than others! This method is
        one attempt, which makes all of the matrices have orthogonal rows or columns, W_O into a
        rotation and W_Q and W_K having the nth column in each having the same norm. The formula is
        $W_V = U @ S,W_O=Vh.T,W_Q=U@S.sqrt(),W_K=Vh@S.sqrt()$.

        More details:

        If W_OV = U @ S @ Vh.T in its singular value decomposition, (where S is in R^d_head not
        R^d_model, as W_OV is low rank), W_OV = (U @ S) @ (Vh.T) is an equivalent low rank
        factorisation, where rows/columns of each matrix are orthogonal! So setting $W_V=US$ and
        $W_O=Vh.T$ works just as well. I *think* this is a more interpretable setup, because now
        $W_O$ is just a rotation, and doesn't change the norm, so $z$ has the same norm as the
        result of the head.

        For $W_QK = W_Q @ W_K.T$ we use the refactor $W_Q = U @ S.sqrt()$ and $W_K = Vh @ S.sqrt()$,
        which is also equivalent ($S==S.sqrt() @ S.sqrt()$ as $S$ is diagonal). Here we keep the
        matrices as having the same norm, since there's not an obvious asymmetry between the keys
        and queries.

        Biases are more fiddly to deal with. For OV it's pretty easy - we just need (x @ W_V + b_V)
        @ W_O + b_O to be preserved, so we can set b_V' = 0. and b_O' = b_V @ W_O + b_O (note that
        b_V in R^{head_index x d_head} while b_O in R^{d_model}, so we need to sum b_V @ W_O along
        the head_index dimension too).

        For QK it's messy - we need to preserve the bilinear form of (x @ W_Q + b_Q) * (y @ W_K +
        b_K), which is fairly messy. To deal with the biases, we concatenate them to W_Q and W_K to
        simulate a d_model+1 dimensional input (whose final coordinate is always 1), do the SVD
        factorization on this effective matrix, then separate out into final weights and biases.
        """

        assert (
            self.cfg.positional_embedding_type != "rotary"
        ), "You can't refactor the QK circuit when using rotary embeddings (as the QK matrix depends on the position of the query and key)"

        for l in range(self.cfg.n_layers):
            # W_QK = W_Q @ W_K.T
            # Concatenate biases to make a d_model+1 input dimension
            W_Q_eff = torch.cat(
                [
                    state_dict[f"blocks.{l}.attn.W_Q"],
                    state_dict[f"blocks.{l}.attn.b_Q"][:, None, :],
                ],
                dim=1,
            )
            W_K_eff = torch.cat(
                [
                    state_dict[f"blocks.{l}.attn.W_K"],
                    state_dict[f"blocks.{l}.attn.b_K"][:, None, :],
                ],
                dim=1,
            )

            W_Q_eff_even, W_K_eff_even_T = (
                FactoredMatrix(W_Q_eff, W_K_eff.transpose(-1, -2)).make_even().pair
            )
            W_K_eff_even = W_K_eff_even_T.transpose(-1, -2)

            state_dict[f"blocks.{l}.attn.W_Q"] = W_Q_eff_even[:, :-1, :]
            state_dict[f"blocks.{l}.attn.b_Q"] = W_Q_eff_even[:, -1, :]
            state_dict[f"blocks.{l}.attn.W_K"] = W_K_eff_even[:, :-1, :]
            state_dict[f"blocks.{l}.attn.b_K"] = W_K_eff_even[:, -1, :]

            # W_OV = W_V @ W_O
            W_V = state_dict[f"blocks.{l}.attn.W_V"]
            W_O = state_dict[f"blocks.{l}.attn.W_O"]

            # Factors the bias to be consistent.
            b_V = state_dict[f"blocks.{l}.attn.b_V"]
            b_O = state_dict[f"blocks.{l}.attn.b_O"]
            effective_bias = b_O + einsum(
                "head_index d_head, head_index d_head d_model -> d_model", b_V, W_O
            )
            state_dict[f"blocks.{l}.attn.b_V"] = torch.zeros_like(b_V)
            state_dict[f"blocks.{l}.attn.b_O"] = effective_bias

            # Helper class to efficiently deal with low rank factored matrices.
            W_OV = FactoredMatrix(W_V, W_O)
            U, S, Vh = W_OV.svd()
            state_dict[f"blocks.{l}.attn.W_V"] = U @ S.diag_embed()
            state_dict[f"blocks.{l}.attn.W_O"] = transpose(Vh)

        return state_dict

    def set_use_attn_result(self, use_attn_result: bool):
        """Toggle whether to explicitly calculate and expose the result for each attention head.

        Useful for interpretability but can easily burn through GPU memory.
        """
        self.cfg.use_attn_result = use_attn_result

    def set_use_split_qkv_input(self, use_split_qkv_input: bool):
        """
        Toggles whether to allow editing of inputs to each attention head.
        """
        self.cfg.use_split_qkv_input = use_split_qkv_input

    def set_use_hook_mlp_in(self, use_hook_mlp_in: bool):
        """Toggles whether to allow storing and editing inputs to each MLP layer."""

        assert not self.cfg.attn_only, "Can't use hook_mlp_in with attn_only model"
        self.cfg.use_hook_mlp_in = use_hook_mlp_in

    def set_use_attn_in(self, use_attn_in: bool):
        """
        Toggles whether to allow editing of inputs to each attention head.
        """
        self.cfg.use_attn_in = use_attn_in

    def check_hooks_to_add(
        self,
        hook_point,
        hook_point_name,
        hook,
        dir="fwd",
        is_permanent=False,
        prepend=False,
    ) -> None:
        if hook_point_name.endswith("attn.hook_result"):
            assert (
                self.cfg.use_attn_result
            ), f"Cannot add hook {hook_point_name} if use_attn_result_hook is False"
        if hook_point_name.endswith(("hook_q_input", "hook_k_input", "hook_v_input")):
            assert (
                self.cfg.use_split_qkv_input
            ), f"Cannot add hook {hook_point_name} if use_split_qkv_input is False"
        if hook_point_name.endswith("mlp_in"):
            assert (
                self.cfg.use_hook_mlp_in
            ), f"Cannot add hook {hook_point_name} if use_hook_mlp_in is False"
        if hook_point_name.endswith("attn_in"):
            assert (
                self.cfg.use_attn_in
            ), f"Cannot add hook {hook_point_name} if use_attn_in is False"

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        is_timm: bool = True,
        is_clip: bool = False,
        fold_ln: Optional[bool] = True,
        center_writing_weights: Optional[bool] = True,
        refactor_factored_attn_matrices: Optional[bool] = False,
        checkpoint_index: Optional[int] = None,
        checkpoint_value: Optional[int] = None,
        hf_model: Optional[ViTForImageClassification] = None,
        device: Optional[Union[str, torch.device]] = None,
        n_devices: Optional[int] = 1,
        move_to_device: Optional[bool] = True,
        fold_value_biases: Optional[bool] = True,
        default_prepend_bos: Optional[bool] = True,
        default_padding_side: Optional[Literal["left", "right"]] = "right",
        dtype="float32",
        use_attn_result: Optional[bool] = False,
        model_type: ModelType = ModelType.VISION,
        **from_pretrained_kwargs,
    ) -> "HookedViT":
        assert not (
                from_pretrained_kwargs.get("load_in_8bit", False)
                or from_pretrained_kwargs.get("load_in_4bit", False)
        ), "Quantization not supported"

        if isinstance(dtype, str):
            # Convert from string to a torch dtype
            dtype = DTYPE_FROM_STRING[dtype]

        if "torch_dtype" in from_pretrained_kwargs:
            # For backwards compatibility with the previous way to do low precision loading
            # This should maybe check the user did not explicitly set dtype *and* torch_dtype
            dtype = from_pretrained_kwargs["torch_dtype"]

        if (
                (from_pretrained_kwargs.get("torch_dtype", None) == torch.float16)
                or dtype == torch.float16
        ) and device in ["cpu", None]:
            logging.warning(
                "float16 models may not work on CPU. Consider using a GPU or bfloat16."
            )

        # Set up other parts of transformer
        cfg = convert_pretrained_model_config(
            model_name,
            is_timm=is_timm,
            is_clip=is_clip,
            model_type=model_type,
        )

        state_dict = get_pretrained_state_dict(
            model_name,
            is_timm,
            is_clip,
            cfg,
            hf_model,
            dtype=dtype,
            return_old_state_dict=True,
            model_type=model_type,
            **from_pretrained_kwargs,
        )

        model = cls(cfg)

        # set false if openclip; not working properly
        if is_clip and model_name.startswith("open-clip"):
            center_writing_weights = False
            print("Setting center_writing_weights to False for OpenCLIP")
            fold_ln = False
            print("Setting fold_ln to False for OpenCLIP")

        model.load_and_process_state_dict(
            state_dict,
            fold_ln=fold_ln,
            center_writing_weights=center_writing_weights,
            fold_value_biases=fold_value_biases,
            refactor_factored_attn_matrices=refactor_factored_attn_matrices,
        )

        # Set up other parameters
        model.set_use_attn_result(use_attn_result)

        print(f"Loaded pretrained model {model_name} into HookedTransformer")

        return model
