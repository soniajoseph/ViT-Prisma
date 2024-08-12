import torch
import torchvision

import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots


from tqdm import tqdm

import einops

import argparse

import random

import numpy as np
import os
import requests

from dataclasses import dataclass
from vit_prisma.sae.config import VisionModelSAERunnerConfig

from vit_prisma.sae.training.activations_store import VisionActivationsStore
# import dataloader
from torch.utils.data import DataLoader

from vit_prisma.utils.data_utils.imagenet_utils import setup_imagenet_paths
from vit_prisma.dataloaders.imagenet_dataset import get_imagenet_transforms_clip, ImageNetValidationDataset
from vit_prisma.models.base_vit import HookedViT

from vit_prisma.sae.sae import SparseAutoencoder

from vit_prisma.dataloaders.imagenet_dataset import get_imagenet_index_to_name


import matplotlib.pyplot as plt

from typing import Any, List, Tuple, Dict

from scipy.stats import gaussian_kde


# import cross-entropy loss
import torch.nn.functional as F
# import partial
from functools import partial

@dataclass
class EvalConfig(VisionModelSAERunnerConfig):
    sae_path: str = '/network/scratch/s/sonia.joseph/sae_checkpoints/1f89d99e-wkcn-TinyCLIP-ViT-40M-32-Text-19M-LAION400M-expansion-16/n_images_520028.pt'
    model_name: str = "wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M"
    model_type: str =  "clip"
    patch_size: str = 32

    dataset_path = "/network/scratch/s/sonia.joseph/datasets/kaggle_datasets"
    dataset_train_path: str = "/network/scratch/s/sonia.joseph/datasets/kaggle_datasets/ILSVRC/Data/CLS-LOC/train"
    dataset_val_path: str = "/network/scratch/s/sonia.joseph/datasets/kaggle_datasets/ILSVRC/Data/CLS-LOC/val"

    verbose: bool = True

    device: bool = 'cuda'

    eval_max: int = 50
    batch_size: int = 32

    @property
    def max_image_output_folder(self) -> str:
        # Get the base directory of sae_checkpoints
        sae_base_dir = os.path.dirname(os.path.dirname(self.sae_path))
        
        # Get the name of the original SAE checkpoint folder
        sae_folder_name = os.path.basename(os.path.dirname(self.sae_path))
        
        # Create a new folder path in sae_checkpoints/images with the original name
        output_folder = os.path.join(sae_base_dir, 'max_images', sae_folder_name)
        output_folder = os.path.join(output_folder, f"layer_{self.hook_point_layer}") # Add layer number

        # Ensure the directory exists
        os.makedirs(output_folder, exist_ok=True)
        
        return output_folder
    
    @property
    def save_figure_dir(self) -> str:
        # Get the base directory of sae_checkpoints
        sae_base_dir = os.path.dirname(os.path.dirname(self.sae_path))
        
        # Get the name of the original SAE checkpoint folder
        sae_folder_name = os.path.basename(os.path.dirname(self.sae_path))
        
        # Create a new folder path in sae_checkpoints/images with the original name
        output_folder = os.path.join(sae_base_dir, 'save_fig_dir', sae_folder_name)
        output_folder = os.path.join(output_folder, f"layer_{self.hook_point_layer}") # Add layer number

        # Ensure the directory exists
        os.makedirs(output_folder, exist_ok=True)
        return output_folder   

def create_eval_config(args):
    return EvalConfig(
        sae_path=args.sae_path,
        model_name=args.model_name,
        model_type=args.model_type,
        patch_size=args.patch_size,
        dataset_path=args.dataset_path,
        dataset_train_path=args.dataset_train_path,
        dataset_val_path=args.dataset_val_path,
        device=args.device,
        verbose=args.verbose,
        eval_max=args.eval_max,
        batch_size=args.batch_size,
        samples_per_bin=args.samples_per_bin,
        max_images_per_feature=args.max_images_per_feature,
        output_folder=args.output_folder
    )

def setup_environment():
    torch.set_grad_enabled(False)

def load_model(cfg):
    from vit_prisma.models.base_vit import HookedViT
    model = HookedViT.from_pretrained(cfg.model_name, is_timm=False, is_clip=True).to(cfg.device)
    model.eval()
    return model

def load_sae(cfg):
    sparse_autoencoder = SparseAutoencoder(cfg).load_from_pretrained(cfg.sae_path)
    sparse_autoencoder.to(cfg.device)
    sparse_autoencoder.eval()  # prevents error if we're expecting a dead neuron mask for who 
    return sparse_autoencoder


def load_dataset(cfg):
    if cfg.model_type == 'clip':
        data_transforms = get_imagenet_transforms_clip(cfg.model_name)
    else:
        raise ValueError("Invalid model type")
    imagenet_paths = setup_imagenet_paths(cfg.dataset_path)
    # train_data = torchvision.datasets.ImageFolder(cfg.dataset_train_path, transform=data_transforms)
    val_data = ImageNetValidationDataset(cfg.dataset_val_path, 
                                    imagenet_paths['label_strings'], 
                                    imagenet_paths['val_labels'], 
                                    data_transforms,
                                    return_index=True,
    )
    val_data_visualize = ImageNetValidationDataset(cfg.dataset_val_path, 
                                    imagenet_paths['label_strings'], 
                                    imagenet_paths['val_labels'],
                                    torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),]), return_index=True)

    print(f"Validation data length: {len(val_data)}") if cfg.verbose else None
    # activations_loader = VisionActivationsStore(cfg, model, train_data, eval_dataset=val_data)
    val_dataloader = DataLoader(val_data, batch_size=cfg.batch_size, shuffle=False, num_workers=4)
    return val_data, val_data_visualize, val_dataloader

def average_l0_test(cfg, val_dataloader, sparse_autoencoder, model, evaluation_max=100):
    total_l0 = []
    with torch.no_grad():
        for i in range(evaluation_max):
            batch_tokens, labels, indices = next(iter(val_dataloader))
            batch_tokens = batch_tokens.to(cfg.device)
            _, cache = model.run_with_cache(batch_tokens, names_filter = sparse_autoencoder.cfg.hook_point)
            sae_out, feature_acts, loss, mse_loss, l1_loss, _ = sparse_autoencoder(
                cache[sparse_autoencoder.cfg.hook_point].to(cfg.device)
            )
            del cache

            # ignore the bos token, get the number of features that activated in each token, averaged accross batch and position
            l0 = (feature_acts[:, :] > 0).float().sum(-1).detach()
            total_l0.append(l0)
    average_l0 = torch.cat(total_l0).mean(0)
    print(f"Average L0: {average_l0.mean()}") if cfg.verbose else None

    # Create histogram using matplotlib
    plt.figure(figsize=(10, 6))
    plt.hist(average_l0.flatten().cpu().numpy(), bins=50, edgecolor='black')
    plt.title("Distribution of Average L0")
    plt.xlabel("Average L0")
    plt.ylabel("Frequency")

    # Save the figure
    save_path = os.path.join(cfg.save_figure_dir, "average_l0.png")
    plt.savefig(save_path)
    plt.close()  # Close the figure to free up memory

    print(f"Saved average l0 figure to {save_path}") if cfg.verbose else None

def get_text_embeddings(model_name, original_text, batch_size=32):
    from transformers import CLIPProcessor, CLIPModel
    vanilla_model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name, do_rescale=False)

    # Split the text into batches
    text_batches = [original_text[i:i+batch_size] for i in range(0, len(original_text), batch_size)]

    all_embeddings = []

    for batch in text_batches:
        inputs = processor(text=batch, return_tensors='pt', padding=True, truncation=True, max_length=77)
        # inputs = {k: v.to(cfg.device) for k, v in inputs.items()}

        with torch.no_grad():
            text_embeddings = vanilla_model.get_text_features(**inputs)

        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        all_embeddings.append(text_embeddings)

    # Concatenate all batches
    final_embeddings = torch.cat(all_embeddings, dim=0)

    return final_embeddings

@torch.no_grad()
def get_recons_loss(
    sparse_autoencoder: SparseAutoencoder,
    model: HookedViT,
    batch_tokens: torch.Tensor,
    gt_labels: torch.Tensor,
    all_labels: List[str],
    text_embeddings: torch.Tensor,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
):
    # Move model to device if it's not already there
    model = model.to(device)
    
    # Move all tensors to the same device
    batch_tokens = batch_tokens.to(device)
    gt_labels = gt_labels.to(device)
    text_embeddings = text_embeddings.to(device)

    # Get image embeddings
    image_embeddings, _ = model.run_with_cache(batch_tokens)

    # Calculate similarity scores
    softmax_values, top_k_indices = get_similarity(image_embeddings, text_embeddings, device=device)

    # Calculate cross-entropy loss
    loss = F.cross_entropy(softmax_values, gt_labels)
    # Safely extract the loss value
    loss_value = loss.item() if torch.isfinite(loss).all() else float('nan')


    head_index = sparse_autoencoder.cfg.hook_point_head_index
    hook_point = sparse_autoencoder.cfg.hook_point

    def standard_replacement_hook(activations: torch.Tensor, hook: Any):
        activations = sparse_autoencoder.forward(activations)[0].to(activations.dtype)
        return activations

    def head_replacement_hook(activations: torch.Tensor, hook: Any):
        new_activations = sparse_autoencoder.forward(activations[:, :, head_index])[0].to(activations.dtype)
        activations[:, :, head_index] = new_activations
        return activations

    replacement_hook = standard_replacement_hook if head_index is None else head_replacement_hook

    recons_image_embeddings = model.run_with_hooks(
        batch_tokens,
        fwd_hooks=[(hook_point, partial(replacement_hook))],
    )
    recons_softmax_values, _ = get_similarity(recons_image_embeddings, text_embeddings, device=device)
    recons_loss = F.cross_entropy(recons_softmax_values, gt_labels)

    zero_abl_image_embeddings = model.run_with_hooks(
        batch_tokens, fwd_hooks=[(hook_point, zero_ablate_hook)]
    )
    zero_abl_softmax_values, _ = get_similarity(zero_abl_image_embeddings, text_embeddings, device=device)
    zero_abl_loss = F.cross_entropy(zero_abl_softmax_values, gt_labels)

    score = (zero_abl_loss - recons_loss) / (zero_abl_loss - loss)

    return score, loss, recons_loss, zero_abl_loss

def get_similarity(image_features, text_features, k=5, device='cuda'):
  image_features = image_features.to(device)
  text_features = text_features.to(device)

  softmax_values = (image_features @ text_features.T).softmax(dim=-1)
  top_k_values, top_k_indices = torch.topk(softmax_values, k, dim=-1)
  return softmax_values, top_k_indices

def get_text_labels(name='wordbank'):
    """
    Loads the library of logit labels from a GitHub URL.

    Returns:
    list: A list of string labels.
    """
    if name == 'wordbank':
        url = "https://raw.githubusercontent.com/yossigandelsman/clip_text_span/main/text_descriptions/image_descriptions_general.txt"
        try:
            # Fetch the content from the URL
            response = requests.get(url)
            response.raise_for_status()  # Raise an exception for bad status codes
            
            # Split the content into lines and strip whitespace
            all_labels = [line.strip() for line in response.text.splitlines()]
            
            print(f"Number of labels loaded: {len(all_labels)}")
            print(f"First 5 labels: {all_labels[:5]}")
            return all_labels
        
        except requests.RequestException as e:
            print(f"An error occurred while fetching the labels: {e}")
            return []
    elif name == 'imagenet':
        from vit_prisma.dataloaders.imagenet_dataset import get_imagenet_text_labels
        return get_imagenet_text_labels()
    else:
        raise ValueError(f"Invalid label set name: {name}")
    
def zero_ablate_hook(activations: torch.Tensor, hook: Any):
    activations = torch.zeros_like(activations)
    return activations

def get_feature_probability(feature_acts):
    return (feature_acts.abs() > 0).float().flatten(0, 1)

def calculate_log_frequencies(total_acts, total_tokens):
    feature_probs = total_acts / total_tokens
    log_feature_probs = torch.log10(feature_probs)
    return log_feature_probs.cpu().numpy()

def process_dataset(model, sparse_autoencoder, dataloader, cfg):
    all_l0 = []
    all_l0_cls = []
    total_loss = 0
    total_reconstruction_loss = 0
    total_zero_abl_loss = 0
    total_samples = 0
    all_cosine_similarity = []

    model.eval()
    sparse_autoencoder.eval()

    all_labels = get_text_labels('imagenet')
    text_embeddings = get_text_embeddings(cfg.model_name, all_labels)

    total_acts = None
    total_tokens = 0

    with torch.no_grad():
        for batch_tokens, gt_labels, indices in tqdm(dataloader):
            batch_tokens = batch_tokens.to(cfg.device)
            batch_size = batch_tokens.shape[0]
            total_samples += 1

            _, cache = model.run_with_cache(batch_tokens, names_filter=sparse_autoencoder.cfg.hook_point)
            hook_point_activation = cache[sparse_autoencoder.cfg.hook_point].to(cfg.device)
            
            sae_out, feature_acts, loss, mse_loss, l1_loss, _ = sparse_autoencoder(hook_point_activation)

            # Calculate feature probability
            sae_activations = get_feature_probability(hook_point_activation)
            if total_acts is None:
                total_acts = sae_activations.sum(0)
            else:
                total_acts += sae_activations.sum(0)
            total_tokens += sae_activations.shape[0]

            # Get L0 stats
            l0 = (feature_acts[:, 1:, :] > 0).float().sum(-1).detach()
            all_l0.extend(l0.mean(dim=1).cpu().numpy())
            l0_cls = (feature_acts[:, 0, :] > 0).float().sum(-1).detach()
            all_l0_cls.extend(l0_cls.flatten().cpu().numpy())

            # Calculate cosine similarity between original activations and sae output
            cos_sim = torch.cosine_similarity(einops.rearrange(hook_point_activation, "batch seq d_mlp -> (batch seq) d_mlp"),
                                                                              einops.rearrange(sae_out, "batch seq d_mlp -> (batch seq) d_mlp"),
                                                                                dim=0).mean(-1).tolist()
            all_cosine_similarity.append(cos_sim)

            # Calculate substitution loss
            score, loss, recons_loss, zero_abl_loss = get_recons_loss(sparse_autoencoder, model, batch_tokens, gt_labels, all_labels, 
                                                                      text_embeddings, device=cfg.device)

            total_loss += loss.item()
            total_reconstruction_loss += recons_loss.item()
            total_zero_abl_loss += zero_abl_loss.item()

            if total_samples >= cfg.eval_max:
                break

    # Calculate average metrics
    avg_loss = total_loss / total_samples
    avg_reconstruction_loss = total_reconstruction_loss / total_samples
    avg_zero_abl_loss = total_zero_abl_loss / total_samples
    avg_l0 = np.mean(all_l0)
    avg_l0_cls = np.mean(all_l0_cls)
    avg_cos_sim = np.mean(all_cosine_similarity)
    log_frequencies = calculate_log_frequencies(total_acts, total_tokens)

    # print out everything above
    print(f"Average L0 (features activated): {avg_l0:.4f}")
    print(f"Average L0 (features activated) per CLS token: {avg_l0_cls:.4f}")
    print(f"Average Cosine Similarity: {avg_cos_sim:.4f}")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Average Reconstruction Loss: {avg_reconstruction_loss:.4f}")
    print(f"Average Zero Ablation Loss: {avg_zero_abl_loss:.4f}")

    return avg_loss, avg_cos_sim, avg_reconstruction_loss, avg_zero_abl_loss, avg_l0, avg_l0_cls, log_frequencies

def plot_log_frequency_histogram(log_frequencies, bins=300):
    plt.figure(figsize=(10, 6))
    plt.hist(log_frequencies, bins=bins, edgecolor='black')
    plt.title("Log Feature Frequency Histogram")
    plt.xlabel("Log10 Feature Frequency")
    plt.ylabel("Count")
    
    # Save as PNG
    plt.savefig("log_frequency_histogram.png", dpi=300, bbox_inches='tight')
    
    # Save as SVG
    plt.savefig("log_frequency_histogram.svg", format='svg', bbox_inches='tight')
    
    plt.close()  

def get_intervals_for_sparsities(log_freq):
    # Define intervals and conditions
    intervals = [
        (-8, -6),
        (-6, -5),
        (-5, -4),
        (-4, -3),
        (-3, -2),
        (-2, -1),
        (-float('inf'), -8),
        (-1, float('inf'))
    ]

    conditions = [torch.logical_and(log_freq >= lower, log_freq < upper) for lower, upper in intervals]
    condition_texts = [f"TOTAL_logfreq_[{lower},{upper}]" for lower, upper in intervals]

    # Replace infinity with appropriate text for readability
    condition_texts[-2] = condition_texts[-2].replace('-inf', '-∞')
    condition_texts[-1] = condition_texts[-1].replace('inf', '∞')
    return intervals, conditions, condition_texts

def visualize_sparsities(log_freq, conditions, condition_texts, name, sparse_autoencoder):
    # Create main figure for log frequency histogram
    plt.figure(figsize=(12, 8))
    plt.hist(log_freq.cpu().numpy(), bins=100, density=True, alpha=0.7, edgecolor='black')
    plt.title(f"{name} Log Frequency of Features")
    plt.xlabel("$log_{10}(freq)$")
    plt.ylabel("Density (%)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f"{name}_frequency_histogram.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{name}_frequency_histogram.svg", format='svg', bbox_inches='tight')
    plt.close()

    for condition, condition_text in zip(conditions, condition_texts):
        percentage = (torch.count_nonzero(condition)/log_freq.shape[0]).item()*100
        if percentage == 0:
            continue
        percentage = int(np.round(percentage))
        
        # Select the encoder directions for the condition
        rare_encoder_directions = sparse_autoencoder.W_enc[condition, :]
        rare_encoder_directions_normalized = rare_encoder_directions / rare_encoder_directions.norm(dim=1, keepdim=True)

        # Compute pairwise cosine similarities
        cos_sims_rare = (rare_encoder_directions_normalized @ rare_encoder_directions_normalized.T).flatten()
        
        # If there are too many similarities, sample randomly
        if cos_sims_rare.shape[0] > 10000:
            cos_sims_rare_random_sample = cos_sims_rare[torch.randint(0, cos_sims_rare.shape[0], (10000,))].cpu().numpy()
        else:
            cos_sims_rare_random_sample = cos_sims_rare.cpu().numpy()


        # Create figure with histogram and box plot
        fig, (ax_hist, ax_box) = plt.subplots(2, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [3, 1]})

        # Histogram
        ax_hist.hist(cos_sims_rare_random_sample, bins=100, density=True, alpha=0.7, edgecolor='black')
        ax_hist.set_title(f"{name} Cosine similarities of random {condition_text} encoder directions\nwith each other ({percentage}% of features)")
        ax_hist.set_xlabel("Cosine similarity")
        ax_hist.set_ylabel("Density (%)")
        ax_hist.grid(True, linestyle='--', alpha=0.7)

        # Add KDE
        kde = gaussian_kde(cos_sims_rare_random_sample)
        x_range = np.linspace(cos_sims_rare_random_sample.min(), cos_sims_rare_random_sample.max(), 100)
        ax_hist.plot(x_range, kde(x_range), 'r-', lw=2)

        # Box plot
        ax_box.boxplot(cos_sims_rare_random_sample, vert=False, widths=0.7)
        ax_box.set_yticks([])
        ax_box.set_xlabel("Cosine similarity")
        ax_box.grid(True, axis='x', linestyle='--', alpha=0.7)

        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(f"{name}_cosine_similarity_{condition_text}.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{name}_cosine_similarity_{condition_text}.svg", format='svg', bbox_inches='tight')
        plt.close()


def sample_features_from_bins(log_freq, conditions, condition_labels, samples_per_bin=50):
    sampled_indices = []
    sampled_values = []
    sampled_bin_labels = []

    for condition, bin_label in zip(conditions, condition_labels):
        potential_indices = torch.nonzero(condition, as_tuple=True)[0]
        
        # If there are fewer potential indices than requested samples, use all available
        if len(potential_indices) <= samples_per_bin:
            selected_indices = potential_indices
        else:
            # Randomly sample indices
            selected_indices = potential_indices[torch.randperm(len(potential_indices))[:samples_per_bin]]
        
        values = log_freq[selected_indices]
        
        sampled_indices.extend(selected_indices.tolist())
        sampled_values.extend(values.tolist())
        sampled_bin_labels.extend([bin_label] * len(selected_indices))

    return sampled_indices, sampled_values, sampled_bin_labels  


def collect_max_activating_images(
    data_loader,
    model,
    sparse_autoencoder,
    feature_indices: List[int],
    feature_categories: List[str],
    device: torch.device,
    max_samples: int,
    batch_size: int,
    k: int = 16
):
    torch.no_grad()

    def highest_activating_tokens(
        images,
        model,
        sparse_autoencoder,
        W_enc,
        b_enc,
        feature_ids: List[int],
        feature_categories: List[str],
        k: int = 10,
    ) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        _, cache = model.run_with_cache(images)
        inp = cache[sparse_autoencoder.cfg.hook_point]
        b, seq_len, _ = inp.shape
        post_reshaped = einops.rearrange(inp, "batch seq d_mlp -> (batch seq) d_mlp")
        sae_in = post_reshaped - sparse_autoencoder.b_dec

        acts = einops.einsum(
            sae_in,
            W_enc,
            "... d_in, d_in n -> ... n",
        )
        
        acts = acts + b_enc
        acts = torch.nn.functional.relu(acts)
        unshape = einops.rearrange(acts, "(batch seq) d_in -> batch seq d_in", batch=b, seq=seq_len)
        cls_acts = unshape[:,0,:]
        per_image_acts = unshape.mean(1)

        to_return = {}
        for i, (feature_id, feature_cat) in enumerate(zip(feature_ids, feature_categories)):
            if "CLS_" in feature_cat:
                top_acts_values, top_acts_indices = cls_acts[:,i].topk(k)
            else:
                top_acts_values, top_acts_indices = per_image_acts[:,i].topk(k)
            to_return[feature_id] = (top_acts_indices, top_acts_values)
        return to_return

    max_indices = {i: None for i in feature_indices}
    max_values = {i: None for i in feature_indices}
    b_enc = sparse_autoencoder.b_enc[feature_indices]
    W_enc = sparse_autoencoder.W_enc[:, feature_indices]

    for batch_idx, (total_images, _, total_indices) in tqdm(enumerate(data_loader), total=max_samples//batch_size):
        total_images = total_images.to(device)
        total_indices = total_indices.to(device)
        new_stuff = highest_activating_tokens(total_images, model, sparse_autoencoder, W_enc, b_enc, feature_indices, feature_categories, k=k)
        
        for feature_id in feature_indices:
            new_indices, new_values = new_stuff[feature_id]
            new_indices = total_indices[new_indices]
            
            if max_indices[feature_id] is None:
                max_indices[feature_id] = new_indices
                max_values[feature_id] = new_values
            else:
                ABvals = torch.cat((max_values[feature_id], new_values))
                ABinds = torch.cat((max_indices[feature_id], new_indices))
                _, inds = torch.topk(ABvals, new_values.shape[0])
                max_values[feature_id] = ABvals[inds]
                max_indices[feature_id] = ABinds[inds]

        if (batch_idx + 1) * batch_size >= max_samples:
            break

    top_per_feature = {i: (max_values[i].detach().cpu(), max_indices[i].detach().cpu()) for i in feature_indices}
    return top_per_feature 


def get_heatmap(image, model, sparse_autoencoder, feature_id, device):
    with torch.no_grad():
        image = image.to(device)
        _, cache = model.run_with_cache(image.unsqueeze(0))
        post_reshaped = einops.rearrange(cache[sparse_autoencoder.cfg.hook_point], "batch seq d_mlp -> (batch seq) d_mlp")
        sae_in = post_reshaped - sparse_autoencoder.b_dec
        acts = einops.einsum(
            sae_in,
            sparse_autoencoder.W_enc[:, feature_id],
            "x d_in, d_in -> x",
        )
    return acts 

def create_patch_heatmap(activation_values, image_size=224, patch_size=16):
    pixel_num = image_size // patch_size
    activation_values = activation_values.detach().cpu().numpy()[1:]
    activation_values = activation_values.reshape(pixel_num, pixel_num)
    heatmap = np.zeros((image_size, image_size))
    for i in range(pixel_num):
        for j in range(pixel_num):
            heatmap[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = activation_values[i, j]
    return heatmap

def prepare_image_data(top_per_feature, feature_id, val_data_visualize, val_data, ind_to_name):
    max_vals, max_inds = top_per_feature[feature_id]
    images, model_images, gt_labels = [], [], []
    for bid, v in zip(max_inds, max_vals):
        image, label, image_ind = val_data_visualize[bid]
        assert image_ind.item() == bid
        images.append(image)
        model_img, _, _ = val_data[bid]
        model_images.append(model_img)
        gt_labels.append(ind_to_name[str(label)][1])
    return images, model_images, gt_labels, max_vals, max_inds

def create_heatmap_plot(images, model_images, gt_labels, max_vals, max_inds, 
                        model, sparse_autoencoder, feature_id, category, cfg):
    grid_size = int(np.ceil(np.sqrt(len(images))))
    fig, axs = plt.subplots(int(np.ceil(len(images)/grid_size)), grid_size, figsize=(15, 15))
    fig.suptitle(f"Category: {category},  Feature: {feature_id}")
    for ax in axs.flatten():
        ax.axis('off')

    complete_bid = []
    for i, (image_tensor, label, val, bid, model_img) in enumerate(zip(images, gt_labels, max_vals, max_inds, model_images)):
        if bid in complete_bid:
            continue 
        complete_bid.append(bid)

        row, col = i // grid_size, i % grid_size
        heatmap = get_heatmap(model_img, model, sparse_autoencoder, feature_id, cfg.device)
        heatmap = create_patch_heatmap(heatmap, patch_size=cfg.patch_size)

        display = image_tensor.numpy().transpose(1, 2, 0)

        axs[row, col].imshow(display)
        axs[row, col].imshow(heatmap, cmap='viridis', alpha=0.3)
        axs[row, col].set_title(f"{label} {val.item():0.03f}")
        axs[row, col].axis('off')

    plt.tight_layout()
    return fig

def save_heatmap_plot(fig, category, logfreq, feature_id, output_folder):
    folder = os.path.join(output_folder, f"{category}")
    os.makedirs(folder, exist_ok=True)
    plt.savefig(os.path.join(folder, f"neglogfreq_{-logfreq}feature_id:{feature_id}.png"))
    plt.close(fig)

def generate_feature_heatmaps(top_per_feature, sampled_indices, sampled_bin_labels, sampled_values,
                              val_data_visualize, val_data, ind_to_name, model, sparse_autoencoder, cfg):
    print("Generating feature heatmaps...")
    for feature_id, category, logfreq in tqdm(zip(sampled_indices, sampled_bin_labels, sampled_values), 
                                              total=len(sampled_bin_labels)):
        images, model_images, gt_labels, max_vals, max_inds = prepare_image_data(
            top_per_feature, feature_id, val_data_visualize, val_data, ind_to_name
        )
        
        fig = create_heatmap_plot(images, model_images, gt_labels, max_vals, max_inds, 
                                  model, sparse_autoencoder, feature_id, category, cfg)
        
        save_heatmap_plot(fig, category, logfreq, feature_id, cfg.max_image_output_folder)


def evaluate():
    cfg = create_eval_config()
    setup_environment()
    model = load_model(cfg)
    sparse_autoencoder = load_sae(cfg)
    val_data, val_data_visualize, val_dataloader = load_dataset(cfg)
    print("Loaded model and data") if cfg.verbose else None

    print("Processing dataset...")
    avg_loss, avg_cos_sim, avg_reconstruction_loss, avg_zero_abl_loss, avg_l0, avg_l0_cls, log_frequencies = process_dataset(model, sparse_autoencoder, val_dataloader, cfg)
    
    print("Plotting log frequencies...")
    plot_log_frequency_histogram(log_frequencies)
    log_freq = torch.Tensor(log_frequencies)
    intervals, conditions, conditions_texts = get_intervals_for_sparsities(log_freq)
    visualize_sparsities(log_freq, conditions, conditions_texts, "TOTAL", sparse_autoencoder)

    print("Getting maximally activating images...")
    sampled_indices, sampled_values, sampled_bin_labels = sample_features_from_bins(
    log_freq=log_freq,
    conditions=conditions,
    condition_labels=conditions_texts,
    samples_per_bin=50
)
    
    top_per_feature = collect_max_activating_images(
    data_loader=val_dataloader,
    model=model,
    sparse_autoencoder=sparse_autoencoder,
    feature_indices=sampled_indices,
    feature_categories=sampled_bin_labels,
    device=cfg.device,
    max_samples=cfg.eval_max,
    batch_size=cfg.batch_size,
    k=16
    )

    print("Plotting heatmaps...")
    ind_to_name = get_imagenet_index_to_name()

    generate_feature_heatmaps(top_per_feature, sampled_indices, sampled_bin_labels, sampled_values,
                              val_data_visualize, val_data, ind_to_name, model, sparse_autoencoder, cfg)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate sparse autoencoder")
    parser.add_argument("--sae_path", type=str, 
                        default='/network/scratch/s/sonia.joseph/sae_checkpoints/1f89d99e-wkcn-TinyCLIP-ViT-40M-32-Text-19M-LAION400M-expansion-16/n_images_520028.pt',
                        help="Path to sparse autoencoder")
    parser.add_argument("--model_name", type=str, 
                        default="wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M",
                        help="Name of the model")
    parser.add_argument("--model_type", type=str, default="clip", help="Type of the model")
    parser.add_argument("--patch_size", type=int, default=32, help="Patch size")
    parser.add_argument("--dataset_path", type=str, 
                        default="/network/scratch/s/sonia.joseph/datasets/kaggle_datasets",
                        help="Path to the dataset")
    parser.add_argument("--dataset_train_path", type=str, 
                        default="/network/scratch/s/sonia.joseph/datasets/kaggle_datasets/ILSVRC/Data/CLS-LOC/train",
                        help="Path to the training dataset")
    parser.add_argument("--dataset_val_path", type=str, 
                        default="/network/scratch/s/sonia.joseph/datasets/kaggle_datasets/ILSVRC/Data/CLS-LOC/val",
                        help="Path to the validation dataset")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--verbose", action="store_true", default=True, help="Verbose output")
    parser.add_argument("--eval_max", type=int, default=50, help="Maximum number of samples to evaluate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--samples_per_bin", type=int, default=50, help="Number of samples per bin")
    parser.add_argument("--max_images_per_feature", type=int, default=16, help="Maximum number of images per feature")
    parser.add_argument("--output_folder", type=str, default="output", help="Output folder")
    
    args = parser.parse_args()
    cfg = create_config(args)
    evaluate(cfg)