import torch
import torchvision
import plotly.express as px
from tqdm import tqdm
import einops
import numpy as np
import os
import requests
from dataclasses import dataclass
from torch.utils.data import DataLoader

# Importing custom modules
from vit_prisma.sae.config import VisionModelSAERunnerConfig
from vit_prisma.models.base_vit import HookedViT
from vit_prisma.utils.data_utils.imagenet_utils import setup_imagenet_paths
from vit_prisma.dataloaders.imagenet_dataset import get_imagenet_transforms_clip, ImageNetValidationDataset
from vit_prisma.sae.sae import SparseAutoencoder

@dataclass
class EvalConfig(VisionModelSAERunnerConfig):
    sae_path: str = 'n_images_130007_log_feature_sparsity.pt'
    model_name: str = "wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M"
    model_type: str =  "clip"
    patch_size: str = 32
    dataset_path: str = "/network/scratch/s/sonia.joseph/datasets/kaggle_datasets"
    dataset_train_path: str = "/network/scratch/s/sonia.joseph/datasets/kaggle_datasets/ILSVRC/Data/CLS-LOC/train"
    dataset_val_path: str = "/network/scratch/s/sonia.joseph/datasets/kaggle_datasets/ILSVRC/Data/CLS-LOC/val"
    verbose: bool = True
    device: str = 'cuda'
    eval_max: int = 50_000
    batch_size: int = 32

    @property
    def max_image_output_folder(self) -> str:
        sae_base_dir = os.path.dirname(os.path.dirname(self.sae_path))
        sae_folder_name = os.path.basename(os.path.dirname(self.sae_path))
        output_folder = os.path.join(sae_base_dir, 'max_images', sae_folder_name, f"layer_{self.hook_point_layer}")
        os.makedirs(output_folder, exist_ok=True)
        return output_folder

def setup_environment():
    cfg = EvalConfig()
    torch.set_grad_enabled(False)
    return cfg

def load_model(cfg):
    model = HookedViT.from_pretrained(cfg.model_name, is_timm=False, is_clip=True).to(cfg.device)
    return model

def load_datasets(cfg):
    data_transforms = get_imagenet_transforms_clip(cfg.model_name)
    imagenet_paths = setup_imagenet_paths(cfg.dataset_path)
    
    train_data = torchvision.datasets.ImageFolder(cfg.dataset_train_path, transform=data_transforms)
    
    val_data = ImageNetValidationDataset(
        cfg.dataset_val_path, 
        imagenet_paths['label_strings'], 
        imagenet_paths['val_labels'], 
        data_transforms,
        return_index=True
    )
    
    val_data_visualize = ImageNetValidationDataset(
        cfg.dataset_val_path, 
        imagenet_paths['label_strings'], 
        imagenet_paths['val_labels'],
        torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
        ]), 
        return_index=True
    )

    if cfg.verbose:
        print(f"Validation data length: {len(val_data)}")

    return train_data, val_data, val_data_visualize

def create_dataloader(dataset, cfg):
    return DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4)

def load_pretrained_sae(cfg):
    sparse_autoencoder = SparseAutoencoder(cfg).load_from_pretrained(cfg.sae_path)
    sparse_autoencoder.to(cfg.device)
    sparse_autoencoder.eval()
    return sparse_autoencoder


def compute_l0_sample(model, sparse_autoencoder, val_dataloader, cfg):
    with torch.no_grad():
        batch_tokens, labels, indices = next(iter(val_dataloader))
        batch_tokens = batch_tokens.to(cfg.device)
        _, cache = model.run_with_cache(batch_tokens, names_filter = sparse_autoencoder.cfg.hook_point)
        sae_out, feature_acts, loss, mse_loss, l1_loss, _ = sparse_autoencoder(
            cache[sparse_autoencoder.cfg.hook_point].to(cfg.device)
        )
        del cache

        l0 = (feature_acts[:, :] > 0).float().sum(-1).detach()
        l0_cls = (feature_acts[:, :] > 0).float().sum(-1).mean(-1).detach()
        print("average l0", l0.mean().item())
        px.histogram(l0.flatten().cpu().numpy()).show()
        
    return l0, l0_cls

@torch.no_grad()
def get_feature_probability(images, model, sparse_autoencoder):
    _, cache = model.run_with_cache(images, names_filter=[sparse_autoencoder.cfg.hook_point])
    sae_out, feature_acts, loss, mse_loss, l1_loss, _ = sparse_autoencoder(
        cache[sparse_autoencoder.cfg.hook_point]
    )
    return (feature_acts.abs() > 0).float().flatten(0, 1)

def process_dataset(val_dataloader, model, sparse_autoencoder, cfg):
    total_acts = None
    total_tokens = 0
    
    for idx, batch in tqdm(enumerate(val_dataloader), total=cfg.eval_max//cfg.batch_size):
        images = batch[0].to(cfg.device)
        sae_activations = get_feature_probability(images, model, sparse_autoencoder)
        
        if total_acts is None:
            total_acts = sae_activations.sum(0)
        else:
            total_acts += sae_activations.sum(0)
        
        total_tokens += sae_activations.shape[0]
        
    return total_acts, total_tokens

def calculate_log_frequencies(total_acts, total_tokens):
    feature_probs = total_acts / total_tokens
    log_feature_probs = torch.log10(feature_probs)
    return log_feature_probs.cpu().numpy()

def plot_histogram_px(log_frequencies, num_bins=100):
    fig = px.histogram(
        x=log_frequencies,
        nbins=num_bins,
        labels={'x': 'Log10 Feature Frequency', 'y': 'Count'},
        title='Log Feature Density Histogram',
        opacity=0.7,
    )
    fig.update_layout(
        bargap=0.1,
        xaxis_title='Log10 Feature Frequency',
        yaxis_title='Count',
        plot_bgcolor='rgba(240, 240, 240, 0.8)',
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='White'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='White'),
    )
    fig.show()

def to_numpy(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, (list, tuple)):
        return np.array(tensor)
    elif isinstance(tensor, (torch.Tensor, torch.nn.parameter.Parameter)):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, (int, float, bool, str)):
        return np.array(tensor)
    else:
        raise ValueError(f"Input to to_numpy has invalid type: {type(tensor)}")

def hist(tensor, save_name, show=True, renderer=None, **kwargs):
    update_layout_set = {"xaxis_range", "yaxis_range", "hovermode", "xaxis_title", "yaxis_title", "colorbar", "colorscale", "coloraxis", "title_x", "bargap", "bargroupgap", "xaxis_tickformat", "yaxis_tickformat", "title_y", "legend_title_text", "xaxis_showgrid", "xaxis_gridwidth", "xaxis_gridcolor", "yaxis_showgrid", "yaxis_gridwidth", "yaxis_gridcolor", "showlegend", "xaxis_tickmode", "yaxis_tickmode", "margin", "xaxis_visible", "yaxis_visible", "bargap", "bargroupgap", "coloraxis_showscale"}
    
    kwargs_post = {k: v for k, v in kwargs.items() if k in update_layout_set}
    kwargs_pre = {k: v for k, v in kwargs.items() if k not in update_layout_set}
    if "bargap" not in kwargs_post:
        kwargs_post["bargap"] = 0.1
    if "margin" in kwargs_post and isinstance(kwargs_post["margin"], int):
        kwargs_post["margin"] = dict.fromkeys(list("tblr"), kwargs_post["margin"])

    histogram_fig = px.histogram(x=to_numpy(tensor), **kwargs_pre)
    histogram_fig.update_layout(**kwargs_post)

    if show:
        px.histogram(x=to_numpy(tensor), **kwargs_pre).update_layout(**kwargs_post).show(renderer)

def visualize_sparsities(log_freq, conditions, condition_texts, name):
    hist(
        log_freq,
        f"{name}_frequency_histogram",
        show=True,
        title=f"{name} Log Frequency of Features",
        labels={"x": "log<sub>10</sub>(freq)"},
        histnorm="percent",
        template="ggplot2"
    )

def get_reconstruction_loss(images, model, autoencoder):
    logits, cache = model.run_with_cache(images, names_filter=[autoencoder.cfg.hook_point])
    sae_out, feature_acts, loss, mse_loss, l1_loss, mse_loss_ghost_resid = autoencoder(
        cache[autoencoder.cfg.hook_point]
    )

    print("Avg L2 norm of acts: ", cache[autoencoder.cfg.hook_point].pow(2).mean().item())
    print("Avg cos sim of neuron reconstructions: ", torch.cosine_similarity(
        einops.rearrange(cache[autoencoder.cfg.hook_point], "batch seq d_mlp -> (batch seq) d_mlp"),
        einops.rearrange(sae_out, "batch seq d_mlp -> (batch seq) d_mlp"),
        dim=0
    ).mean(-1).tolist())
    print("l1", l1_loss.sum().item())
    return mse_loss.item()

def evaluate_sae(model, sparse_autoencoder, val_dataloader, val_data_visualize, cfg):
    print("Computing activations...")
    all_acts, all_labels, all_indices = compute_activations(model, val_dataloader, cfg)
    
    print("Computing feature activations...")
    feature_acts = compute_feature_acts(sparse_autoencoder, all_acts, cfg)
    
    print("Computing statistics...")
    means, stds, max_vals, max_indices = compute_statistics(feature_acts)
    
    print("Plotting feature statistics...")
    plot_feature_statistics(means, stds)
    
    print("Saving max activating images...")
    save_max_activating_images(max_indices, val_data_visualize, cfg)
    
    print("Computing L0 sample...")
    l0, l0_cls = compute_l0_sample(model, sparse_autoencoder, val_dataloader, cfg)
    
    print("Processing dataset for feature probability...")
    total_acts, total_tokens = process_dataset(val_dataloader, model, sparse_autoencoder, cfg)
    
    print("Calculating log frequencies...")
    log_frequencies = calculate_log_frequencies(total_acts, total_tokens)
    
    print("Plotting log frequency histogram...")
    plot_histogram_px(log_frequencies, num_bins=240)
    
    print("Visualizing sparsities...")
    intervals = [
        (-8, -6), (-6, -5), (-5, -4), (-4, -3), (-3, -2), (-2, -1),
        (-float('inf'), -8), (-1, float('inf'))
    ]
    conditions = [torch.logical_and(torch.tensor(log_frequencies) >= lower, torch.tensor(log_frequencies) < upper) for lower, upper in intervals]
    condition_texts = [f"TOTAL_logfreq_[{lower},{upper}]" for lower, upper in intervals]
    condition_texts[-2] = condition_texts[-2].replace('-inf', '-∞')
    condition_texts[-1] = condition_texts[-1].replace('inf', '∞')
    visualize_sparsities(log_frequencies, conditions, condition_texts, "TOTAL")
    
    print("Computing reconstruction loss...")
    this_max = 4
    for batch_idx, (total_images, total_labels, total_indices) in enumerate(val_dataloader):
        total_images = total_images.to(cfg.device)
        reconstruction_loss = get_reconstruction_loss(total_images, model, sparse_autoencoder)
        print("mse", reconstruction_loss)
        if batch_idx >= this_max:
            break
    
    # Computing substitution loss
    print("Computing substitution loss...")

    # Get maximally activating images
    print("Getting maximally activating images...")
    
    return all_acts, all_labels, all_indices, feature_acts, means, stds, max_vals, max_indices, l0, l0_cls, log_frequencies

def main():
    cfg = setup_environment()
    model = load_model(cfg)
    train_data, val_data, val_data_visualize = load_datasets(cfg)
    val_dataloader = create_dataloader(val_data, cfg)
    sparse_autoencoder = load_pretrained_sae(cfg)

    results = evaluate_sae(model, sparse_autoencoder, val_dataloader, val_data_visualize, cfg)

    # Add your evaluation code here
    # For example:
    # evaluate_sae(model, sparse_autoencoder, val_dataloader, cfg)

if __name__ == "__main__":
    main()


# from functools import partial
# from typing import Any, cast

# import pandas as pd
# import torch
# import wandb
# from tqdm import tqdm

# from vit_prisma.prisma_tools.hooked_root_module import HookedRootModule
# from vit_prisma.prisma_tools.hook_point import HookPoint

# from vit_prisma.utils.prisma_utils import get_act_name

# from vit_prisma.sae.training.activations_store import VisionActivationsStore
# from vit_prisma.sae.sae import SparseAutoencoder

# from vit_prisma.models.base_vit import HookedViT

# import torch.nn.functional as F


# @torch.no_grad()
# # similar to run_evals for language but adapted slightly for vision. 
# def run_evals_vision(
#     sparse_autoencoder: SparseAutoencoder,
#     activation_store: VisionActivationsStore,
#     model: HookedViT,
#     n_training_steps: int,
#     suffix: str = "",
# ):
#     hook_point = sparse_autoencoder.cfg.hook_point
#     hook_point_layer = sparse_autoencoder.cfg.hook_point_layer
#     hook_point_head_index = sparse_autoencoder.cfg.hook_point_head_index

#     ### Evals
#     eval_tokens = activation_store.get_batch_tokens()

#     # Get Reconstruction Score
#     # losses_df = recons_loss_batched(
#     #     sparse_autoencoder,
#     #     model,
#     #     activation_store,
#     #     n_batches=10,
#     # )


#     # recons_score = losses_df["score"].mean()
#     # ntp_loss = losses_df["loss"].mean()
#     # recons_loss = losses_df["recons_loss"].mean()
#     # zero_abl_loss = losses_df["zero_abl_loss"].mean()

#     # get cache
#     _, cache = model.run_with_cache(
#         eval_tokens,
#         names_filter=[get_act_name("pattern", hook_point_layer), hook_point],
#     )

#     # get act
#     if sparse_autoencoder.cfg.hook_point_head_index is not None:
#         original_act = cache[sparse_autoencoder.cfg.hook_point][
#             :, :, sparse_autoencoder.cfg.hook_point_head_index
#         ]
#     else:
#         original_act = cache[sparse_autoencoder.cfg.hook_point]

#     sae_out, _feature_acts, _, _, _, _ = sparse_autoencoder(original_act)
#     patterns_original = (
#         cache[get_act_name("pattern", hook_point_layer)][:, hook_point_head_index]
#         .detach()
#         .cpu()
#     )
#     del cache

#     if "cuda" in str(model.cfg.device):
#         torch.cuda.empty_cache()

#     l2_norm_in = torch.norm(original_act, dim=-1)
#     l2_norm_out = torch.norm(sae_out, dim=-1)
#     l2_norm_ratio = l2_norm_out / l2_norm_in

#     wandb.log(
#         {
#             # l2 norms
#             f"metrics/l2_norm{suffix}": l2_norm_out.mean().item(),
#             f"metrics/l2_ratio{suffix}": l2_norm_ratio.mean().item(),
#             # CE Loss
#             # f"metrics/CE_loss_score{suffix}": recons_score,
#             # f"metrics/ce_loss_without_sae{suffix}": ntp_loss,
#             # f"metrics/ce_loss_with_sae{suffix}": recons_loss,
#             # f"metrics/ce_loss_with_ablation{suffix}": zero_abl_loss,
#         },
#         step=n_training_steps,
#     )

#     head_index = sparse_autoencoder.cfg.hook_point_head_index

#     def standard_replacement_hook(activations: torch.Tensor, hook: Any):
#         activations = sparse_autoencoder.forward(activations)[0].to(activations.dtype)
#         return activations

#     def head_replacement_hook(activations: torch.Tensor, hook: Any):
#         new_actions = sparse_autoencoder.forward(activations[:, :, head_index])[0].to(
#             activations.dtype
#         )
#         activations[:, :, head_index] = new_actions
#         return activations

#     head_index = sparse_autoencoder.cfg.hook_point_head_index
#     replacement_hook = (
#         standard_replacement_hook if head_index is None else head_replacement_hook
#     )

#     # get attn when using reconstructed activations
#     with model.hooks(fwd_hooks=[(hook_point, partial(replacement_hook))]):
#         _, new_cache = model.run_with_cache(
#             eval_tokens, names_filter=[get_act_name("pattern", hook_point_layer)]
#         )
#         patterns_reconstructed = (
#             new_cache[get_act_name("pattern", hook_point_layer)][
#                 :, hook_point_head_index
#             ]
#             .detach()
#             .cpu()
#         )
#         del new_cache

#     # get attn when using reconstructed activations
#     with model.hooks(fwd_hooks=[(hook_point, partial(zero_ablate_hook))]):
#         _, zero_ablation_cache = model.run_with_cache(
#             eval_tokens, names_filter=[get_act_name("pattern", hook_point_layer)]
#         )
#         patterns_ablation = (
#             zero_ablation_cache[get_act_name("pattern", hook_point_layer)][
#                 :, hook_point_head_index
#             ]
#             .detach()
#             .cpu()
#         )
#         del zero_ablation_cache

#     if sparse_autoencoder.cfg.hook_point_head_index:
#         kl_result_reconstructed = kl_divergence_attention(
#             patterns_original, patterns_reconstructed
#         )
#         kl_result_reconstructed = kl_result_reconstructed.sum(dim=-1).numpy()

#         kl_result_ablation = kl_divergence_attention(
#             patterns_original, patterns_ablation
#         )
#         kl_result_ablation = kl_result_ablation.sum(dim=-1).numpy()

#         if wandb.run is not None:
#             wandb.log(
#                 {
#                     f"metrics/kldiv_reconstructed{suffix}": kl_result_reconstructed.mean().item(),
#                     f"metrics/kldiv_ablation{suffix}": kl_result_ablation.mean().item(),
#                 },
#                 step=n_training_steps,
#             )

# def recons_loss_batched(
#     sparse_autoencoder: SparseAutoencoder,
#     model: HookedViT,
#     activation_store: VisionActivationsStore,
#     n_batches: int = 100,
# ):
#     losses = []
#     for _ in tqdm(range(n_batches)):
#         batch_tokens, labels = activation_store.get_val_batch_tokens()
        
#         score, loss, recons_loss, zero_abl_loss = get_recons_loss(
#             sparse_autoencoder, model, batch_tokens, labels,
#         )
#         losses.append(
#             (
#                 score.mean().item(),
#                 loss.mean().item(),
#                 recons_loss.mean().item(),
#                 zero_abl_loss.mean().item(),
#             )
#         )

#     losses = pd.DataFrame(
#         losses, columns=cast(Any, ["score", "loss", "recons_loss", "zero_abl_loss"])
#     )

#     return losses


# @torch.no_grad()
# def get_recons_loss(
#     sparse_autoencoder: SparseAutoencoder,
#     model: HookedViT,
#     batch_tokens: torch.Tensor,
#     labels:torch.Tensor
# ):
#     hook_point = sparse_autoencoder.cfg.hook_point
#     class_logits = model(batch_tokens)
#     loss = F.cross_entropy(class_logits, labels)



#     head_index = sparse_autoencoder.cfg.hook_point_head_index

#     def standard_replacement_hook(activations: torch.Tensor, hook: Any):
#         activations = sparse_autoencoder.forward(activations)[0].to(activations.dtype)
#         return activations

#     def head_replacement_hook(activations: torch.Tensor, hook: Any):
#         new_activations = sparse_autoencoder.forward(activations[:, :, head_index])[
#             0
#         ].to(activations.dtype)
#         activations[:, :, head_index] = new_activations
#         return activations

#     replacement_hook = (
#         standard_replacement_hook if head_index is None else head_replacement_hook
#     )
#     recons_class_logits = model.run_with_hooks(
#         batch_tokens,
#         fwd_hooks=[(hook_point, partial(replacement_hook))],
#     )
#     recons_loss = F.cross_entropy(recons_class_logits, labels)

#     zero_abl_class_logits = model.run_with_hooks(
#         batch_tokens, fwd_hooks=[(hook_point, zero_ablate_hook)]
#     )
#     zero_abl_loss = F.cross_entropy(zero_abl_class_logits, labels)

#     score = (zero_abl_loss - recons_loss) / (zero_abl_loss - loss)

#     return score, loss, recons_loss, zero_abl_loss


# def zero_ablate_hook(activations: torch.Tensor, hook: Any):
#     activations = torch.zeros_like(activations)
#     return activations


# def kl_divergence_attention(y_true: torch.Tensor, y_pred: torch.Tensor):
#     # Compute log probabilities for KL divergence
#     log_y_true = torch.log2(y_true + 1e-10)
#     log_y_pred = torch.log2(y_pred + 1e-10)

#     return y_true * (log_y_true - log_y_pred)