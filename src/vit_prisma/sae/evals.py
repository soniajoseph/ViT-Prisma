import torch
import torchvision

import plotly.express as px

from tqdm import tqdm

import einops

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

import matplotlib.pyplot as plt

from typing import Any, List

# import cross-entropy loss
import torch.nn.functional as F
# import partial
from functools import partial



def create_eval_config():
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

        eval_max: int = 1 # 50_000
        batch_size: int = 32

        # make the max image output folder a subfolder of the sae path


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
        

    cfg = EvalConfig()
    return cfg

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

    with torch.no_grad():
        for batch_tokens, gt_labels, indices in tqdm(dataloader):
            batch_tokens = batch_tokens.to(cfg.device)
            batch_size = batch_tokens.shape[0]
            total_samples += 1

            _, cache = model.run_with_cache(batch_tokens, names_filter=sparse_autoencoder.cfg.hook_point)
            hook_point_activation = cache[sparse_autoencoder.cfg.hook_point].to(cfg.device)
            
            sae_out, feature_acts, loss, mse_loss, l1_loss, _ = sparse_autoencoder(hook_point_activation)

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

    # print out everything above
    print(f"Average L0 (features activated): {avg_l0:.4f}")
    print(f"Average L0 (features activated) per CLS token: {avg_l0_cls:.4f}")
    print(f"Average Cosine Similarity: {avg_cos_sim:.4f}")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Average Reconstruction Loss: {avg_reconstruction_loss:.4f}")
    print(f"Average Zero Ablation Loss: {avg_zero_abl_loss:.4f}")

    return avg_loss, avg_cos_sim, avg_reconstruction_loss, avg_zero_abl_loss, avg_l0, avg_l0_cls


def evaluate():

    cfg = create_eval_config()
    setup_environment()
    model = load_model(cfg)
    sparse_autoencoder = load_sae(cfg)
    val_data, val_data_visualize, val_dataloader = load_dataset(cfg)
    print("Loaded model and data") if cfg.verbose else None

    print("Processing dataset...")

    avg_loss, avg_cos_sim, avg_reconstruction_loss, avg_zero_abl_loss, avg_l0, avg_l0_cls = process_dataset(model, sparse_autoencoder, val_dataloader, cfg)

    # Create and save the histogram
    # Create and save the histogram of activated features per token
    fig_activated = px.histogram(avg_l0, title="Distribution of Activated Features per Token")
    fig_activated.update_layout(
        xaxis_title="Number of Activated Features",
        yaxis_title="Count"
    )
    fig_activated.write_image("histogram_activated_features.svg")

    # Create and save the histogram of avg activated features per sample
    fig_cls = px.histogram(avg_l0_cls, title="Distribution of Avg Activated Features per CLS token")
    fig_cls.update_layout(
        xaxis_title="Average Number of Activated Features",
        yaxis_title="Count"
    )
    fig_cls.write_image("histogram_activated_features_cls.svg")

    # Create and save the histogram of feature sparsity
    fig_sparsity = px.histogram(feature_sparsity.cpu().numpy(), title="Distribution of Feature Sparsity")
    fig_sparsity.update_layout(
        xaxis_title="Feature Sparsity",
        yaxis_title="Count"
    )
    fig_sparsity.write_image("histogram_feature_sparsity.svg")

    # Create and save a box plot of feature sparsity
    fig_box = go.Figure()
    fig_box.add_trace(go.Box(y=feature_sparsity.cpu().numpy(), name="Feature Sparsity"))
    fig_box.update_layout(
        title="Box Plot of Feature Sparsity",
        yaxis_title="Feature Sparsity"
    )
    fig_box.write_image("boxplot_feature_sparsity.svg")

    print("Plots saved as 'histogram_activated_features.svg', 'histogram_avg_activated_features.svg', 'histogram_feature_sparsity.svg', and 'boxplot_feature_sparsity.svg'")

if __name__ == '__main__':
    evaluate()

    