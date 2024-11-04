import einops
import torch
import torchvision


import pickle

from dataclasses import dataclass
from vit_prisma.sae.config import VisionModelSAERunnerConfig

@dataclass
class EvalConfig(VisionModelSAERunnerConfig):
    sae_path: str = '/workspace/sae_checkpoints/8e32860c-clip-b-sae-gated-all-tokens-x64-layer-9-mlp-out-v1/n_images_260014.pt'
    model_name: str = "open-clip:laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K"
    model_type: str =  "clip"
    patch_size: str = 32

    dataset_path = "/workspace"
    dataset_train_path: str = "/workspace/ILSVRC/Data/CLS-LOC/train"
    dataset_val_path: str = "/workspace/ILSVRC/Data/CLS-LOC/val"

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

cfg = EvalConfig()

import importlib
import vit_prisma
# importlib.reload(vit_prisma.dataloaders.imagenet_dataset)

# load dataset
import open_clip
from vit_prisma.utils.data_utils.imagenet_utils import setup_imagenet_paths
from vit_prisma.dataloaders.imagenet_dataset import get_imagenet_transforms_clip, ImageNetValidationDataset

from torchvision import transforms
from transformers import CLIPProcessor

og_model_name = "hf-hub:laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K"
og_model, _, preproc = open_clip.create_model_and_transforms(og_model_name)
processor = preproc

size=224

data_transforms = transforms.Compose([
    transforms.Resize((size, size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                     std=[0.26862954, 0.26130258, 0.27577711]),
])
    
imagenet_paths = setup_imagenet_paths(cfg.dataset_path)
imagenet_paths["train"] = "/workspace/ILSVRC/Data/CLS-LOC/train"
imagenet_paths["val"] = "/workspace/ILSVRC/Data/CLS-LOC/val"
imagenet_paths["val_labels"] = "/workspace/LOC_val_solution.csv"
imagenet_paths["label_strings"] = "/workspace/LOC_synset_mapping.txt"
print()
train_data = torchvision.datasets.ImageFolder(cfg.dataset_train_path, transform=data_transforms)
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





interesting_features_category = pickle.load(open("/workspace/interesting_features_category_oct_31.pkl", "rb"))
interesting_features_values = pickle.load(open("/workspace/interesting_features_values_oct_31.pkl", "rb"))
top_activations_per_feature = pickle.load(open("/workspace/top_activations_per_feature_oct_31.pkl", "rb"))

from vit_prisma.dataloaders.imagenet_dataset import get_imagenet_index_to_name
ind_to_name = get_imagenet_index_to_name()


import matplotlib.pyplot as plt
import torch
import plotly.express as px

from tqdm import tqdm

import numpy as np
import os

torch.no_grad()
def get_heatmap(
          image,
          model,
          sparse_autoencoder,
          feature_id,
): 
    image = image.to(cfg.device)
    _, cache = model.run_with_cache(image.unsqueeze(0))

    post_reshaped = einops.rearrange(cache[sparse_autoencoder.cfg.hook_point], "batch seq d_mlp -> (batch seq) d_mlp")
    # Compute activations (not from a fwd pass, but explicitly, by taking only the feature we want)
    # This code is copied from the first part of the 'forward' method of the AutoEncoder class
    sae_in =  post_reshaped - sparse_autoencoder.b_dec # Remove decoder bias as per Anthropic
    acts = einops.einsum(
            sae_in,
            sparse_autoencoder.W_enc[:, feature_id],
            "x d_in, d_in -> x",
        )
    return acts 
     
def image_patch_heatmap(activation_values,image_size=224, pixel_num=14):
    activation_values = activation_values.detach().cpu().numpy()
    activation_values = activation_values[1:]
    activation_values = activation_values.reshape(pixel_num, pixel_num)

    # Create a heatmap overlay
    heatmap = np.zeros((image_size, image_size))
    patch_size = image_size // pixel_num

    for i in range(pixel_num):
        for j in range(pixel_num):
            heatmap[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = activation_values[i, j]

    return heatmap

    # Removing axes


for feature_ids, cat, logfreq in tqdm(zip(top_activations_per_feature.keys(), interesting_features_category, interesting_features_values), total=len(interesting_features_category)):
    # need to output the images in a subfolder, not a figure with them
    max_vals, max_inds = top_activations_per_feature[feature_ids]
    images = []
    model_images = []
    gt_labels = []
    for bid, v in zip(max_inds, max_vals):
        image, label, image_ind = val_data_visualize[bid]

        assert image_ind.item() == bid
        images.append(image)

        # model_img, _, _ = imagenet_data[bid]
        model_image, _, _ = val_data[bid]
        # I think we're looking for model_image??
        model_images.append(model_image)
        gt_labels.append(ind_to_name[str(label)][1])

#         print(len(images))
#         grid_size = int(np.ceil(np.sqrt(len(images))))
#         fig, axs = plt.subplots(int(np.ceil(len(images)/grid_size)), grid_size, figsize=(15, 15))
#         name=  f"Category: {cat},  Feature: {feature_ids}"
#         fig.suptitle(name)#, y=0.95)
#         for ax in axs.flatten():
#             ax.axis('off')
    complete_bid = []

    folder = os.path.join(cfg.max_image_output_folder, f"{cat}")
    os.makedirs(folder, exist_ok=True)
    for i, (image_tensor, label, val, bid,model_img) in enumerate(zip(images, gt_labels, max_vals,max_inds,model_images )):
        if bid in complete_bid:
            continue 
        complete_bid.append(bid)

        # print(type(image_tensor))
        newlabel = label.replace("'", "_").replace("-", "_")
        # print(f"neglogfreq_{-logfreq}feauture_id:{feature_ids}label:{newlabel}bid:{bid}.png")
        savethis = image_tensor.numpy().transpose(1, 2, 0)
        # print(type(savethis))
        plt.imsave(os.path.join(folder, f"neglogfreq_{-logfreq}feauture_id:{feature_ids}label:{newlabel}bid:{bid}.png"), savethis)


    plt.close()


