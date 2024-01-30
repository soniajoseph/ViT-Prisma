"""
Reference:
https://github.com/neelnanda-io/TransformerLens/blob/main/transformer_lens/ActivationCache.py
"""

from typing import Dict, Iterator, List, Optional, Tuple, Union, Float
import torch
import logging


class ActivationCache:

    def __init__(
            self, cache_dict: Dict[str, torch.Tensor], model, has_batch_dim: bool = True
    ):
        self.cache_dict = cache_dict
        self.model = model
        self.has_batch_dim = has_batch_dim
        self.has_embed = "hook_embed" in self.cache_dict
        self.has_pos_embed = "hook_pos_embed" in self.cache_dict

    def remove_batch_dim(self) -> ActivationCache:
        """Remove the Batch Dimension (if a single batch item).

        Returns:
            The ActivationCache with the batch dimension removed.
        """
        if self.has_batch_dim:
            for key in self.cache_dict:
                assert (
                    self.cache_dict[key].size(0) == 1
                ), f"Cannot remove batch dimension from cache with batch size > 1, \
                    for key {key} with shape {self.cache_dict[key].shape}"
                self.cache_dict[key] = self.cache_dict[key][0]
            self.has_batch_dim = False
        else:
            logging.warning(
                "Tried removing batch dimension after already having removed it."
            )
        return self

    def __repr__(self) -> str:
        """Representation of the ActivationCache.

        Special method that returns a string representation of an object. It's normally used to give
        a string that can be used to recreate the object, but here we just return a string that
        describes the object.
        """
        return f"ActivationCache with keys {list(self.cache_dict.keys())}"        
    

    def __getitem__(self, key) -> torch.Tensor:
        """Retrieve Cached Activations by Key or Shorthand.

        Enables direct access to cached activations via dictionary-style indexing using keys or
        shorthand naming conventions. It also supports tuples for advanced indexing, with the
        dimension order as (get_act_name, layer_index, layer_type).

        Args:
            key:
                The key or shorthand name for the activation to retrieve.

        Returns:
            The cached activation tensor corresponding to the given key.
        """
        if key in self.cache_dict:
            return self.cache_dict[key]
        elif type(key) == str:
            return self.cache_dict[utils.get_act_name(key)]
        else:
            if len(key) > 1 and key[1] is not None:
                if key[1] < 0:
                    # Supports negative indexing on the layer dimension
                    key = (key[0], self.model.cfg.n_layers + key[1], *key[2:])
            return self.cache_dict[utils.get_act_name(*key)]
    
    def __len__(self) -> int:
        """Length of the ActivationCache.

        Special method that returns the length of an object (in this case the number of different
        activations in the cache).
        """
        return len(self.cache_dict)

    def keys(self):
            """Keys of the ActivationCache.

            Examples:

                >>> from transformer_lens import HookedTransformer
                >>> model = HookedTransformer.from_pretrained("tiny-stories-1M")
                Loaded pretrained model tiny-stories-1M into HookedTransformer
                >>> _logits, cache = model.run_with_cache("Some prompt")
                >>> list(cache.keys())[0:3]
                ['hook_embed', 'hook_pos_embed', 'blocks.0.hook_resid_pre']

            Returns:
                List of all keys.
            """
            return self.cache_dict.keys()

    def values(self):
            """Values of the ActivationCache.

            Returns:
                List of all values.
            """
            return self.cache_dict.values()

    def items(self):
        """Items of the ActivationCache.

        Returns:
            List of all items ((key, value) tuples).
        """
        return self.cache_dict.items()

    def __iter__(self) -> Iterator[Tuple[str, torch.Tensor]]:
        """ActivationCache Iterator.

        Special method that returns an iterator over the ActivationCache. Allows looping over the
        cache.

        Examples:

            >>> from transformer_lens import HookedTransformer
            >>> model = HookedTransformer.from_pretrained("tiny-stories-1M")
            Loaded pretrained model tiny-stories-1M into HookedTransformer
            >>> _logits, cache = model.run_with_cache("Some prompt")
            >>> cache_interesting_names = []
            >>> for key in cache:
            ...     if not key.startswith("blocks.") or key.startswith("blocks.0"):
            ...         cache_interesting_names.append(key)
            >>> print(cache_interesting_names[0:3])
            ['hook_embed', 'hook_pos_embed', 'blocks.0.hook_resid_pre']

        Returns:
            Iterator over the cache.
        """
        return self.cache_dict.__iter__()
    
    def accumulated_resid(
        self,
        layer: Optional[int] = None,
        incl_mid: Optional[bool] = False,
        apply_ln: Optional[bool] = False,
        pos_slice: Optional[Union[Slice, SliceInput]] = None,
        mlp_input: Optional[bool] = False,
        return_labels: Optional[bool] = False,
    ) -> Union[
        Float[torch.Tensor, "layers_covered *batch_and_pos_dims d_model"],
        Tuple[
            Float[torch.Tensor, "layers_covered *batch_and_pos_dims d_model"], List[str]
        ],
    ]:
        """Accumulated Residual Stream.

        Returns the accumulated residual stream at each layer/sub-layer. This is useful for `Logit
        Lens <https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens>`
        style analysis, where it can be thought of as what the model "believes" at each point in the
        residual stream.

        To project this into the vocabulary space, remember that there is a final layer norm in most
        decoder-only transformers. Therefore, you need to first apply the final layer norm (which
        can be done with `apply_ln`), and then multiply by the unembedding matrix (:math:`W_U`).

        If you instead want to look at contributions to the residual stream from each component
        (e.g. for direct logit attribution), see :meth:`decompose_resid` instead, or
        :meth:`get_full_resid_decomposition` if you want contributions broken down further into each
        MLP neuron.

        Examples:

        Logit Lens analysis can be done as follows:

        >>> from transformer_lens import HookedTransformer
        >>> from einops import einsum
        >>> import torch
        >>> import pandas as pd

        >>> model = HookedTransformer.from_pretrained("tiny-stories-1M", device="cpu")
        Loaded pretrained model tiny-stories-1M into HookedTransformer

        >>> prompt = "Why did the chicken cross the"
        >>> answer = " road"
        >>> logits, cache = model.run_with_cache("Why did the chicken cross the")
        >>> answer_token = model.to_single_token(answer)
        >>> print(answer_token)
        2975

        >>> accum_resid, labels = cache.accumulated_resid(return_labels=True, apply_ln=True)
        >>> last_token_accum = accum_resid[:, 0, -1, :]  # layer, batch, pos, d_model
        >>> print(last_token_accum.shape)  # layer, d_model
        torch.Size([9, 64])

        >>> W_U = model.W_U
        >>> print(W_U.shape)
        torch.Size([64, 50257])

        >>> layers_unembedded = einsum(
        ...         last_token_accum,
        ...         W_U,
        ...         "layer d_model, d_model d_vocab -> layer d_vocab"
        ...     )
        >>> print(layers_unembedded.shape)
        torch.Size([9, 50257])

        >>> # Get the rank of the correct answer by layer
        >>> sorted_indices = torch.argsort(layers_unembedded, dim=1, descending=True)
        >>> rank_answer = (sorted_indices == 2975).nonzero(as_tuple=True)[1]
        >>> print(pd.Series(rank_answer, index=labels))
        0_pre         4442
        1_pre          382
        2_pre          982
        3_pre         1160
        4_pre          408
        5_pre          145
        6_pre           78
        7_pre          387
        final_post       6
        dtype: int64

        Args:
            layer:
                The layer to take components up to - by default includes resid_pre for that layer
                and excludes resid_mid and resid_post for that layer. If set as `n_layers`, `-1` or
                `None` it will return all residual streams, including the final one (i.e.
                immediately pre logits). The indices are taken such that this gives the accumulated
                streams up to the input to layer l.
            incl_mid:
                Whether to return `resid_mid` for all previous layers.
            apply_ln:
                Whether to apply LayerNorm to the stack.
            pos_slice:
                A slice object to apply to the pos dimension. Defaults to None, do nothing.
            mlp_input:
                Whether to include resid_mid for the current layer. This essentially gives the MLP
                input rather than the attention input.
            return_labels:
                Whether to return a list of labels for the residual stream components. Useful for
                labelling graphs.

        Returns:
            A tensor of the accumulated residual streams. If `return_labels` is True, also returns a
            list of labels for the components (as a tuple in the form `(components, labels)`).
        """
        if not isinstance(pos_slice, Slice):
            pos_slice = Slice(pos_slice)
        if layer is None or layer == -1:
            # Default to the residual stream immediately pre unembed
            layer = self.model.cfg.n_layers
        assert isinstance(layer, int)
        labels = []
        components = []
        for l in range(layer + 1):
            if l == self.model.cfg.n_layers:
                components.append(self[("resid_post", self.model.cfg.n_layers - 1)])
                labels.append("final_post")
                continue
            components.append(self[("resid_pre", l)])
            labels.append(f"{l}_pre")
            if (incl_mid and l < layer) or (mlp_input and l == layer):
                components.append(self[("resid_mid", l)])
                labels.append(f"{l}_mid")
        components = [pos_slice.apply(c, dim=-2) for c in components]
        components = torch.stack(components, dim=0)
        if apply_ln:
            components = self.apply_ln_to_stack(
                components, layer, pos_slice=pos_slice, mlp_input=mlp_input
            )
        if return_labels:
            return components, labels
        else:
            return components
        
    