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

from vit_prisma.sae.sae import SparseAutoencoder


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

        eval_max: int = 50_000 # 50_000
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
    px.histogram(average_l0.flatten().cpu().numpy()).show()
    fig.write_image(os.path.join(cfg.save_figure_dir,"average_l0.png"))  # Save as PNG
    print(f"Saved average l0 figure to {cfg.save_figure_dir}") if cfg.verbose else None

import torch
import plotly.express as px
import numpy as np

def process_dataset(model, sparse_autoencoder, dataloader, cfg):
    all_l0 = []
    all_l0_cls = []
    total_loss = 0
    total_mse_loss = 0
    total_l1_loss = 0
    total_samples = 0

    model.eval()
    sparse_autoencoder.eval()

    with torch.no_grad():
        for batch_tokens, labels, indices in dataloader:
            batch_tokens = batch_tokens.to(cfg.device)
            batch_size = batch_tokens.size(0)
            total_samples += batch_size

            _, cache = model.run_with_cache(batch_tokens, names_filter=sparse_autoencoder.cfg.hook_point)
            sae_out, feature_acts, loss, mse_loss, l1_loss, _ = sparse_autoencoder(
                cache[sparse_autoencoder.cfg.hook_point].to(cfg.device)
            )

            # Ignore the bos token, get the number of features that activated in each token
            l0 = (feature_acts[:, :] > 0).float().sum(-1).detach()
            l0_cls = l0.mean(-1).detach()

            all_l0.extend(l0.flatten().cpu().numpy())
            all_l0_cls.extend(l0_cls.cpu().numpy())

            total_loss += loss.item() * batch_size
            total_mse_loss += mse_loss.item() * batch_size
            total_l1_loss += l1_loss.item() * batch_size

    # Calculate average metrics
    avg_loss = total_loss / total_samples
    avg_mse_loss = total_mse_loss / total_samples
    avg_l1_loss = total_l1_loss / total_samples
    avg_l0 = np.mean(all_l0)

    return all_l0, all_l0_cls, avg_loss, avg_mse_loss, avg_l1_loss, avg_l0


import torch
import plotly.express as px
import numpy as np

def process_dataset(model, sparse_autoencoder, dataloader, cfg):
    all_l0 = []
    all_l0_cls = []
    total_loss = 0
    total_mse_loss = 0
    total_l1_loss = 0
    total_substitution_loss = 0
    total_reconstruction_loss = 0
    total_samples = 0

    model.eval()
    sparse_autoencoder.eval()

    with torch.no_grad():
        for batch_tokens, labels, indices in dataloader:
            batch_tokens = batch_tokens.to(cfg.device)
            batch_size = batch_tokens.size(0)
            total_samples += batch_size

            _, cache = model.run_with_cache(batch_tokens, names_filter=sparse_autoencoder.cfg.hook_point)
            hook_point_activation = cache[sparse_autoencoder.cfg.hook_point].to(cfg.device)
            
            sae_out, feature_acts, loss, mse_loss, l1_loss, _ = sparse_autoencoder(hook_point_activation)

            # Calculate substitution loss
            substitution_loss = torch.norm(sae_out - hook_point_activation) / torch.norm(hook_point_activation)
            
            # Calculate reconstruction loss
            reconstruction_loss = torch.nn.functional.mse_loss(sae_out, hook_point_activation)

            # Ignore the bos token, get the number of features that activated in each token
            l0 = (feature_acts[:, :] > 0).float().sum(-1).detach()
            l0_cls = l0.mean(-1).detach()

            all_l0.extend(l0.flatten().cpu().numpy())
            all_l0_cls.extend(l0_cls.cpu().numpy())

            total_loss += loss.item() * batch_size
            total_mse_loss += mse_loss.item() * batch_size
            total_l1_loss += l1_loss.item() * batch_size
            total_substitution_loss += substitution_loss.item() * batch_size
            total_reconstruction_loss += reconstruction_loss.item() * batch_size

    # Calculate average metrics
    avg_loss = total_loss / total_samples
    avg_mse_loss = total_mse_loss / total_samples
    avg_l1_loss = total_l1_loss / total_samples
    avg_substitution_loss = total_substitution_loss / total_samples
    avg_reconstruction_loss = total_reconstruction_loss / total_samples
    avg_l0 = np.mean(all_l0)

    return all_l0, all_l0_cls, avg_loss, avg_mse_loss, avg_l1_loss, avg_substitution_loss, avg_reconstruction_loss, avg_l0




def evaluate():

    cfg = create_eval_config()
    setup_environment()
    model = load_model(cfg)
    sparse_autoencoder = load_sae(cfg)
    val_data, val_data_visualize, val_dataloader = load_dataset(cfg)
    print("Loaded model and data") if cfg.verbose else None

    print("Running average l0 test") if cfg.verbose else None
    average_l0_test(cfg, val_dataloader, sparse_autoencoder, model, evaluation_max=1)

    print("Processing dataset...")

    all_l0, all_l0_cls, avg_loss, avg_mse_loss, avg_l1_loss, avg_l0 = process_dataset(model, sparse_autoencoder, val_dataloader, cfg)

    # Print the results
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Average MSE Loss: {avg_mse_loss:.4f}")
    print(f"Average L1 Loss: {avg_l1_loss:.4f}")
    print(f"Average L0 (features activated): {avg_l0:.4f}")

    # Create and save the histogram
    fig = px.histogram(all_l0, title="Distribution of Activated Features per Token")
    fig.update_layout(
        xaxis_title="Number of Activated Features",
        yaxis_title="Count"
    )
    fig.write_image("histogram_activated_features.svg")

    # Create and save the CLS token histogram
    fig_cls = px.histogram(all_l0_cls, title="Distribution of Avg Activated Features per Sample")
    fig_cls.update_layout(
        xaxis_title="Average Number of Activated Features",
        yaxis_title="Count"
    )
    fig_cls.write_image("histogram_avg_activated_features.svg")

    print("Histograms saved as 'histogram_activated_features.svg' and 'histogram_avg_activated_features.svg'")

    all_l0, all_l0_cls, avg_loss, avg_mse_loss, avg_l1_loss, avg_substitution_loss, avg_reconstruction_loss, avg_l0 = process_dataset(model, sparse_autoencoder, val_dataloader, cfg)

    # Print the results
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Average MSE Loss: {avg_mse_loss:.4f}")
    print(f"Average L1 Loss: {avg_l1_loss:.4f}")
    print(f"Average Substitution Loss: {avg_substitution_loss:.4f}")
    print(f"Average Reconstruction Loss: {avg_reconstruction_loss:.4f}")
    print(f"Average L0 (features activated): {avg_l0:.4f}")

    # Create and save the histogram
    fig = px.histogram(all_l0, title="Distribution of Activated Features per Token")
    fig.update_layout(
        xaxis_title="Number of Activated Features",
        yaxis_title="Count"
    )
    fig.write_image("histogram_activated_features.svg")

    # Create and save the CLS token histogram
    fig_cls = px.histogram(all_l0_cls, title="Distribution of Avg Activated Features per Sample")
    fig_cls.update_layout(
        xaxis_title="Average Number of Activated Features",
        yaxis_title="Count"
    )
    fig_cls.write_image("histogram_avg_activated_features.svg")

    print("Histograms saved as 'histogram_activated_features.svg' and 'histogram_avg_activated_features.svg'")

if __name__ == '__main__':
    evaluate()

    