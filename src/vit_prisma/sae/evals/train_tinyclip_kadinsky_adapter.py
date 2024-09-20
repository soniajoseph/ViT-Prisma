import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPVisionModelWithProjection
from transformers import CLIPModel, CLIPProcessor


from diffusers import KandinskyV22PriorPipeline, KandinskyV22Pipeline
import torchvision.transforms as transforms
from torchvision.datasets import ImageNet
from tqdm import tqdm

from vit_prisma.transforms.open_clip_transforms import get_clip_val_transforms

import torchvision

# import F.normalize
import torch.nn.functional as F

# import CLIPModel
from transformers import CLIPModel

import os

import wandb


from vit_prisma.models.base_vit import HookedViT

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


import random
import string
random_hash = ''.join(random.choices(string.ascii_lowercase + string.digits, k=5))



# Assuming you have a TinyClip model implementation

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin

    def forward(self, adapted_embeddings, target_embeddings):
        # Normalize embeddings
        adapted_embeddings = F.normalize(adapted_embeddings, p=2, dim=1)
        target_embeddings = F.normalize(target_embeddings, p=2, dim=1)

        # Compute cosine similarity for all pairs
        similarity_matrix = torch.matmul(adapted_embeddings, target_embeddings.T)

        # Extract positive pairs (diagonal elements assuming each adapted corresponds to each target)
        positive_pairs = torch.diag(similarity_matrix)

        # Mask to zero-out positive pair impacts in the similarity matrix for negative comparison
        mask = torch.eye(similarity_matrix.size(0), device=similarity_matrix.device, dtype=torch.bool)
        similarity_matrix.masked_fill_(mask, float('-inf'))

        # Get the maximum negative similarity for each positive pair
        max_negative_similarity = torch.max(similarity_matrix, dim=1)[0]
        
        # Compute loss as max(0, margin - positive + max_negative)
        losses = F.relu(self.margin - positive_pairs + max_negative_similarity)

        return losses.mean()

class EmbeddingAdapter(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=2048, output_dim=1280, num_layers=4, dropout_rate=0.1):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(input_dim, hidden_dim)])
        self.layers.extend([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 2)])
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.activation(x)
            x = self.norm(x)
            x = self.dropout(x)
        x = self.layers[-1](x)
        return x

class DualEmbedder:
    def __init__(self, clip_model_name, kandinsky_model):
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.kandinsky_model = kandinsky_model
        
        # Move models to the appropriate device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model.to(self.device)
        self.kandinsky_model.to(self.device)
        
    def get_embeddings(self, images):
        with torch.no_grad():
            # Process images for CLIP
            # clip_inputs = self.clip_processor(images=images, return_tensors="pt").to(self.device)
            
            # Get CLIP embeddings
            clip_outputs = self.clip_model.get_image_features(pixel_values=images).float()
            clip_embeddings = clip_outputs / clip_outputs.norm(dim=-1, keepdim=True)

            
            # Get Kandinsky embeddings
            kandinsky_embeddings = self.kandinsky_model(images).image_embeds

        
        return clip_embeddings, kandinsky_embeddings

# Training function
def train_adapter(adapter, dataloader, dual_embedder, num_epochs=2):
    optimizer = optim.Adam(adapter.parameters(), lr=1e-4)
    criterion = ContrastiveLoss()
    
    global_step = 0
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (images, _) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            images = images.to(DEVICE)
            
            # Get embeddings
            tinyclip_embed, kandinsky_embed = dual_embedder.get_embeddings(images)
            
            optimizer.zero_grad()
            output = adapter(tinyclip_embed)
            loss = criterion(output, kandinsky_embed)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            global_step += 1
            
            # Log every 100 batches
            if global_step % 10 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(dataloader)}, Loss: {avg_loss:.4f}")
                
                # Log to wandb
                wandb.log({
                    "epoch": epoch + 1,
                    "batch": batch_idx + 1,
                    "global_step": global_step,
                    "loss": avg_loss
                })

                        # save checkpoint
                print("Saving checkpoint at global step", global_step)
                save_dir = f'/network/scratch/s/sonia.joseph/diffusion/tinyclip_adapter/{random_hash}'
                # make directory
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                torch.save(adapter.state_dict(), os.path.join(save_dir, f"adapter_checkpoint_{global_step}.pth"))
            

        # Log epoch summary
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.f}")
        wandb.log({
            "epoch": epoch + 1,
            "epoch_avg_loss": avg_loss
        })

# Kandinsky loading function
def load_kandinsky_encoder(cache_dir, device='cuda'):
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        'kandinsky-community/kandinsky-2-2-prior',
        subfolder='image_encoder',
        cache_dir=cache_dir,
    ).to(device)

    # unet = UNet2DConditionModel.from_pretrained(

    #     'kandinsky-community/kandinsky-2-2-decoder',
    #     subfolder='unet',
    #     cache_dir=cache_dir,
    # ).half().to(DEVICE)

    # prior = KandinskyV22PriorPipeline.from_pretrained(
    #     'kandinsky-community/kandinsky-2-2-prior',
    #     image_encoder=image_encoder,
    #     torch_dtype=torch.float16,
    #     cache_dir=cache_dir,
    # ).to(DEVICE)

    # decoder = KandinskyV22Pipeline.from_pretrained(
    #     'kandinsky-community/kandinsky-2-2-decoder',
    #     unet=unet,
    #     torch_dtype=torch.float16,
    #     cache_dir=cache_dir,
    # ).to(DEVICE)

    # zero_embed = prior.get_zero_embed()

    return image_encoder


def load_imagenet(data_transforms, dataset_path, model_type='clip'):
    # Imagenet-specific logic
    from vit_prisma.utils.data_utils.imagenet_utils import setup_imagenet_paths
    from vit_prisma.transforms.open_clip_transforms import get_clip_val_transforms
    from vit_prisma.dataloaders.imagenet_dataset import ImageNetValidationDataset
    if model_type == 'clip':
        data_transforms = get_clip_val_transforms()
    else:
        raise ValueError("Invalid model type")
    imagenet_paths = setup_imagenet_paths(dataset_path)
    train_data = torchvision.datasets.ImageFolder(imagenet_paths['train'], transform=data_transforms)
    val_data = ImageNetValidationDataset(imagenet_paths['val'], 
                                    imagenet_paths['label_strings'], 
                                    imagenet_paths['val_labels'], 
                                    data_transforms
    )
    return train_data, val_data

def load_pretrained_adapter(checkpoint_path, input_dim=512, hidden_dim=2048, output_dim=1280):
    # Initialize the adapter model
    adapter = EmbeddingAdapter(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    
    # Load the state dict
    state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    
    # Load the state dict into the model
    adapter.load_state_dict(state_dict)
    
    # Move the model to the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adapter = adapter.to(device)
    
    # Set the model to evaluation mode
    adapter.eval()
    
    return adapter

def get_imagenet_dataloaders(train_data, val_data, batch_size=32):
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_dataloader, val_dataloader

# Main script
if __name__ == "__main__":

    # generate random hash for name
    import argparse
    parser = argparse.ArgumentParser()
    # pretrained checkpoint
    parser.add_argument('--pretrained_checkpoint', type=str, default=None, required=False)
    args = parser.parse_args()


    device= 'cuda'

    wandb.init(project="tinyclip-kandinsky-adapter", name=f"{random_hash}-adapter-training")

    # Load ImageNet dataset
    clip_transform = get_clip_val_transforms()

    cache_dir = '/network/scratch/s/sonia.joseph/.cache'
    imagenet_path = '/network/scratch/s/sonia.joseph/datasets/kaggle_datasets'

    train_data, val_data = load_imagenet(clip_transform, imagenet_path)
    train_dataloader, val_dataloader = get_imagenet_dataloaders(train_data, val_data, batch_size=256)

    print("Data loaded successfully.")
    
    # Initialize TinyClip and Kandinsky models
    model_name = "wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M"
    tinyclip_model = HookedViT.from_pretrained(model_name, is_timm=False, is_clip=True).to(device)
    tinyclip_model.eval()

    
    kandinsky_encoder = load_kandinsky_encoder(cache_dir, device)
    kandinsky_encoder.eval()

    print("Models loaded successfully.")
    
    # Create activations store
    dual_embedder = DualEmbedder(model_name, kandinsky_encoder)

    # tinyclip_embeddings, kandinsky_embeddings = dual_embedder.get_embeddings(torch.randn(2, 3, 224, 224).to(device))

    # Initialize adapter with larger hidden layer
    if args.pretrained_checkpoint:
        adapter = load_pretrained_adapter(args.pretrained_checkpoint)
    else:
        adapter = EmbeddingAdapter(input_dim=512, hidden_dim=2048, output_dim=1280).to(device)

    # Train adapter
    train_adapter(adapter, train_dataloader, dual_embedder)

    # Save adapter
    save_path = '/network/scratch/s/sonia.joseph/diffusion'
    torch.save(adapter.state_dict(), os.path.join(save_path, 'tinyclip_to_kandinsky_adapter.pth'))

    wandb.finish()
