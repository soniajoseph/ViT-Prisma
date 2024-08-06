from functools import partial
from typing import Any, cast

import pandas as pd
import torch
import wandb
from tqdm import tqdm

from vit_prisma.prisma_tools.hooked_root_module import HookedRootModule
from vit_prisma.prisma_tools.hook_point import HookPoint

from vit_prisma.utils.prisma_utils import get_act_name

from vit_prisma.sae.training.activations_store import VisionActivationsStore
from vit_prisma.sae.sae import SparseAutoencoder

from vit_prisma.models.base_vit import HookedViT

import torch.nn.functional as F


@torch.no_grad()
# similar to run_evals for language but adapted slightly for vision. 
def run_evals_vision(
    sparse_autoencoder: SparseAutoencoder,
    activation_store: VisionActivationsStore,
    model: HookedViT,
    n_training_steps: int,
    suffix: str = "",
):
    hook_point = sparse_autoencoder.cfg.hook_point
    hook_point_layer = sparse_autoencoder.cfg.hook_point_layer
    hook_point_head_index = sparse_autoencoder.cfg.hook_point_head_index

    ### Evals
    eval_tokens = activation_store.get_batch_tokens()

    # Get Reconstruction Score
    # losses_df = recons_loss_batched(
    #     sparse_autoencoder,
    #     model,
    #     activation_store,
    #     n_batches=10,
    # )


    # recons_score = losses_df["score"].mean()
    # ntp_loss = losses_df["loss"].mean()
    # recons_loss = losses_df["recons_loss"].mean()
    # zero_abl_loss = losses_df["zero_abl_loss"].mean()

    # get cache
    _, cache = model.run_with_cache(
        eval_tokens,
        names_filter=[get_act_name("pattern", hook_point_layer), hook_point],
    )

    # get act
    if sparse_autoencoder.cfg.hook_point_head_index is not None:
        original_act = cache[sparse_autoencoder.cfg.hook_point][
            :, :, sparse_autoencoder.cfg.hook_point_head_index
        ]
    else:
        original_act = cache[sparse_autoencoder.cfg.hook_point]

    sae_out, _feature_acts, _, _, _, _ = sparse_autoencoder(original_act)
    patterns_original = (
        cache[get_act_name("pattern", hook_point_layer)][:, hook_point_head_index]
        .detach()
        .cpu()
    )
    del cache

    if "cuda" in str(model.cfg.device):
        torch.cuda.empty_cache()

    l2_norm_in = torch.norm(original_act, dim=-1)
    l2_norm_out = torch.norm(sae_out, dim=-1)
    l2_norm_ratio = l2_norm_out / l2_norm_in

    wandb.log(
        {
            # l2 norms
            f"metrics/l2_norm{suffix}": l2_norm_out.mean().item(),
            f"metrics/l2_ratio{suffix}": l2_norm_ratio.mean().item(),
            # CE Loss
            # f"metrics/CE_loss_score{suffix}": recons_score,
            # f"metrics/ce_loss_without_sae{suffix}": ntp_loss,
            # f"metrics/ce_loss_with_sae{suffix}": recons_loss,
            # f"metrics/ce_loss_with_ablation{suffix}": zero_abl_loss,
        },
        step=n_training_steps,
    )

    head_index = sparse_autoencoder.cfg.hook_point_head_index

    def standard_replacement_hook(activations: torch.Tensor, hook: Any):
        activations = sparse_autoencoder.forward(activations)[0].to(activations.dtype)
        return activations

    def head_replacement_hook(activations: torch.Tensor, hook: Any):
        new_actions = sparse_autoencoder.forward(activations[:, :, head_index])[0].to(
            activations.dtype
        )
        activations[:, :, head_index] = new_actions
        return activations

    head_index = sparse_autoencoder.cfg.hook_point_head_index
    replacement_hook = (
        standard_replacement_hook if head_index is None else head_replacement_hook
    )

    # get attn when using reconstructed activations
    with model.hooks(fwd_hooks=[(hook_point, partial(replacement_hook))]):
        _, new_cache = model.run_with_cache(
            eval_tokens, names_filter=[get_act_name("pattern", hook_point_layer)]
        )
        patterns_reconstructed = (
            new_cache[get_act_name("pattern", hook_point_layer)][
                :, hook_point_head_index
            ]
            .detach()
            .cpu()
        )
        del new_cache

    # get attn when using reconstructed activations
    with model.hooks(fwd_hooks=[(hook_point, partial(zero_ablate_hook))]):
        _, zero_ablation_cache = model.run_with_cache(
            eval_tokens, names_filter=[get_act_name("pattern", hook_point_layer)]
        )
        patterns_ablation = (
            zero_ablation_cache[get_act_name("pattern", hook_point_layer)][
                :, hook_point_head_index
            ]
            .detach()
            .cpu()
        )
        del zero_ablation_cache

    if sparse_autoencoder.cfg.hook_point_head_index:
        kl_result_reconstructed = kl_divergence_attention(
            patterns_original, patterns_reconstructed
        )
        kl_result_reconstructed = kl_result_reconstructed.sum(dim=-1).numpy()

        kl_result_ablation = kl_divergence_attention(
            patterns_original, patterns_ablation
        )
        kl_result_ablation = kl_result_ablation.sum(dim=-1).numpy()

        if wandb.run is not None:
            wandb.log(
                {
                    f"metrics/kldiv_reconstructed{suffix}": kl_result_reconstructed.mean().item(),
                    f"metrics/kldiv_ablation{suffix}": kl_result_ablation.mean().item(),
                },
                step=n_training_steps,
            )

def recons_loss_batched(
    sparse_autoencoder: SparseAutoencoder,
    model: HookedViT,
    activation_store: VisionActivationsStore,
    n_batches: int = 100,
):
    losses = []
    for _ in tqdm(range(n_batches)):
        batch_tokens, labels = activation_store.get_val_batch_tokens()
        
        score, loss, recons_loss, zero_abl_loss = get_recons_loss(
            sparse_autoencoder, model, batch_tokens, labels,
        )
        losses.append(
            (
                score.mean().item(),
                loss.mean().item(),
                recons_loss.mean().item(),
                zero_abl_loss.mean().item(),
            )
        )

    losses = pd.DataFrame(
        losses, columns=cast(Any, ["score", "loss", "recons_loss", "zero_abl_loss"])
    )

    return losses


@torch.no_grad()
def get_recons_loss(
    sparse_autoencoder: SparseAutoencoder,
    model: HookedViT,
    batch_tokens: torch.Tensor,
    labels:torch.Tensor
):
    hook_point = sparse_autoencoder.cfg.hook_point
    class_logits = model(batch_tokens)
    loss = F.cross_entropy(class_logits, labels)



    head_index = sparse_autoencoder.cfg.hook_point_head_index

    def standard_replacement_hook(activations: torch.Tensor, hook: Any):
        activations = sparse_autoencoder.forward(activations)[0].to(activations.dtype)
        return activations

    def head_replacement_hook(activations: torch.Tensor, hook: Any):
        new_activations = sparse_autoencoder.forward(activations[:, :, head_index])[
            0
        ].to(activations.dtype)
        activations[:, :, head_index] = new_activations
        return activations

    replacement_hook = (
        standard_replacement_hook if head_index is None else head_replacement_hook
    )
    recons_class_logits = model.run_with_hooks(
        batch_tokens,
        fwd_hooks=[(hook_point, partial(replacement_hook))],
    )
    recons_loss = F.cross_entropy(recons_class_logits, labels)

    zero_abl_class_logits = model.run_with_hooks(
        batch_tokens, fwd_hooks=[(hook_point, zero_ablate_hook)]
    )
    zero_abl_loss = F.cross_entropy(zero_abl_class_logits, labels)

    score = (zero_abl_loss - recons_loss) / (zero_abl_loss - loss)

    return score, loss, recons_loss, zero_abl_loss


def zero_ablate_hook(activations: torch.Tensor, hook: Any):
    activations = torch.zeros_like(activations)
    return activations


def kl_divergence_attention(y_true: torch.Tensor, y_pred: torch.Tensor):
    # Compute log probabilities for KL divergence
    log_y_true = torch.log2(y_true + 1e-10)
    log_y_pred = torch.log2(y_pred + 1e-10)

    return y_true * (log_y_true - log_y_pred)