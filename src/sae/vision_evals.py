from functools import partial
from typing import Any, cast

import pandas as pd
import torch
import wandb
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name

from sae.vision_activations_store import VisionActivationsStore
from sae_lens.training.sparse_autoencoder import SparseAutoencoder
from sae_lens.training.evals import zero_ablate_hook, kl_divergence_attention
import torch.nn.functional as F


from vit_prisma.models.base_vit import HookedViT
from transformers import CLIPProcessor, CLIPModel
import torchvision
import os

# Load ImageNet folder
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms

import torch

import csv

import numpy as np

import einops

from typing import Optional, Any



@torch.no_grad()
# similar to run_evals for language but adapted slightly for vision. #TODO adapt for CLIP model!
def run_evals_vision(
    sparse_autoencoder: SparseAutoencoder,
    activation_store: VisionActivationsStore,
    model: HookedTransformer,
    n_training_steps: int,
    suffix: str = "",
    is_clip: bool = True,
    precomputed_text_embeddings: Optional[Any] = None, # For CLIP
): 
    hook_point = sparse_autoencoder.cfg.hook_point
    hook_point_layer = sparse_autoencoder.cfg.hook_point_layer
    hook_point_head_index = sparse_autoencoder.cfg.hook_point_head_index

    ### Evals
    eval_tokens = activation_store.get_batch_tokens()

    print("Length of eval tokens in current run_evals_vision: ", len(eval_tokens))

    # if is_clip:
    #     # Get Reconstruction Cross-Entropy Score
    #     recons_score, orig_loss, recons_loss, zero_abl_loss = get_recons_loss_clip(sparse_autoencoder, model,
    #                                                                                eval_tokens, precomputed_clip_text_embeddings=precomputed_text_embeddings)

    #TODO this was set up with classifier vit in mind but hasn't been modified for clip
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

def get_all_stats(
    sparse_autoencoder: SparseAutoencoder,
    model: HookedTransformer,
    activation_store: VisionActivationsStore,
    n_batches: int = 100,
    is_clip: bool = True,
    precomputed_clip_text_embeddings: Optional[Any] = None,
    logit_scale: Optional[Any] = None,
):
    reconstruction_losses = []
    l0_norms = []
    l1_losses = []
    mse_losses = []
    avg_cosine_similarities = []


    sparse_autoencoder = sparse_autoencoder.cuda()

    # for param in sparse_autoencoder.parameters():
    #     param.data = param.data.cuda()
    # model = model.to("cuda")
    # model.cfg = model.cfg.to(model.cfg.device)

    for _ in tqdm(range(n_batches)):

        batch_tokens, labels = activation_store.get_val_batch_tokens()

        hook_point = sparse_autoencoder.cfg.hook_point
        clean_output, cache = model.run_with_cache(batch_tokens, names_filter = [hook_point])
        orig_act = cache[hook_point]
        # orig_act.to(model.cfg.device)
        del cache

        # # L0 Norm
        # sparse_autoencoder = sparse_autoencoder.to(model.cfg.device)

        # sparse_autoencoder.b_dec = sparse_autoencoder.b_dec.to(model.cfg.device)

        sae_out, feature_acts, loss, mse_loss, l1_loss, _ = sparse_autoencoder(
        orig_act
    )

        # ignore the bos token, get the number of features that activated in each token, averaged accross batch and position
        l0_norm = (feature_acts[:, 1:] > 0).float().sum(-1).detach().mean().cpu().item()
        l0_norms.append(l0_norm)

        l1_losses.append(l1_loss)

        mse_losses.append(mse_loss)

        avg_cosine_similarity = torch.cosine_similarity(einops.rearrange(orig_act, "batch seq d_mlp -> (batch seq) d_mlp"),
                                                                              einops.rearrange( sae_out, "batch seq d_mlp -> (batch seq) d_mlp"),
                                                                                dim=0).mean(-1).cpu().tolist()
        avg_cosine_similarities.append(avg_cosine_similarity)
        # Get l1


        # Reconstruction score
        if is_clip:
            score, loss, recons_loss, zero_abl_loss = get_recons_loss_clip(
                orig_act, clean_output,
                sparse_autoencoder, model, batch_tokens, labels, precomputed_clip_text_embeddings, logit_scale
            )
        
        else:
            score, loss, recons_loss, zero_abl_loss = get_recons_loss(
                sparse_autoencoder, model, batch_tokens, labels,
            )
        reconstruction_losses.append(
            (
                score.mean().cpu().item(),
                loss.mean().cpu().item(),
                recons_loss.mean().cpu().item(),
                zero_abl_loss.mean().cpu().item(),
            )
        )

    reconstruction_losses = pd.DataFrame(
        reconstruction_losses, columns=cast(Any, ["score", "loss", "recons_loss", "zero_abl_loss"])



    )

    # Check device of everything below
    l0_norms = torch.tensor(l0_norms).cpu().numpy()
    l1_losses = torch.tensor(l1_losses).cpu().numpy()
    mse_losses = torch.tensor(mse_losses).cpu().numpy()
    avg_cosine_similarities = torch.tensor(avg_cosine_similarities).cpu().numpy()


    # Add everything else to dataframe
    total_stats = pd.DataFrame(
        {
            "l0_norm": l0_norms,
            "l1_loss": l1_losses,
            "mse_loss": mse_losses,
            "avg_cosine_similarity": avg_cosine_similarities,
        }
    )

    # Add reconstruction loss columns
    total_stats = pd.concat([total_stats, reconstruction_losses], axis=1)

    print(total_stats)
    

    return total_stats



def get_logits_from_output_emb(image_emb, text_emb, logit_scale):
    logits_per_text = torch.matmul(text_emb, image_emb.t().to(text_emb.device)) * logit_scale.to(
                text_emb.device
    )
    logits_per_image = logits_per_text.t()
    return logits_per_image



@torch.no_grad()
def get_recons_loss_clip(
    orig_act,
    clean_output,
    sparse_autoencoder: SparseAutoencoder,
    model: HookedTransformer,
    batch_tokens: torch.Tensor,
    labels:torch.Tensor,
    precomputed_clip_text_embeddings: None,
    logit_scale: None,

):

    reconstructed_act = sparse_autoencoder.forward(orig_act).sae_out

    def hook_function(activations, hook, new_activations):
        activations[:] = new_activations
        return activations
    
    hook_point = sparse_autoencoder.cfg.hook_point
    print('hook_point', hook_point)
    
    output_reconstructed = model.run_with_hooks(
        batch_tokens,
        fwd_hooks=[(hook_point, partial(hook_function, new_activations=reconstructed_act))],
    )
    output_zero_ablation = model.run_with_hooks(
        batch_tokens,
        fwd_hooks=[(hook_point, partial(hook_function, new_activations=0.0))],
    )

    # labels = labels.to(model.cfg.device)

    # Get logits
    logits_clean = get_logits_from_output_emb(clean_output, precomputed_clip_text_embeddings, logit_scale).to(clean_output.device)
    loss_clean = F.cross_entropy(logits_clean, labels)

    logits_recon = get_logits_from_output_emb(output_reconstructed, precomputed_clip_text_embeddings, logit_scale).to(clean_output.device)
    loss_reconstructed = F.cross_entropy(logits_recon, labels)

    logits_zero = get_logits_from_output_emb(output_zero_ablation, precomputed_clip_text_embeddings, logit_scale).to(clean_output.device)
    loss_zero = F.cross_entropy(logits_zero, labels)

    percent_reconstructed_score = (loss_zero - loss_reconstructed)/(loss_zero - loss_clean)

    return percent_reconstructed_score, loss_clean, loss_reconstructed, loss_zero



@torch.no_grad()
def get_recons_loss(
    sparse_autoencoder: SparseAutoencoder,
    model: HookedTransformer,
    batch_tokens: torch.Tensor,
    labels: torch.Tensor,
    use_precomputed_clip_text_embeddings: Optional[torch.Tensor] = None,
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