"""
Code to save logits of all imagenet datapoints for future analysis.

"""

import timm
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import h5py
import numpy as np
import logging
import torch.nn.functional as F

from tqdm.auto import tqdm

import os

import argparse

from vit_prisma.circuits.timmAblation import timmAblation

def main(imagenet_path, save_dir, run_all = False, layer_idx=None, head_idx=None, skip_exists=None):
    # Setup basic logging
    logging.basicConfig(level=logging.INFO)

    # 1. Load and preprocess the ImageNet dataset
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
   
    imagenet_data = datasets.ImageFolder(imagenet_path, tr ansform=data_transforms)
    data_loader = DataLoader(imagenet_data, batch_size=64, shuffle=False)
    logging.info(f"ImageNet dataset loaded and preprocessed. Total datapoints: {len(imagenet_data)}")

    # cycle through all attn and mlp ablations and rerun

    # 2. Load a pre-trained model from timm
    model_name = 'vit_base_patch32_224'

    model = timmAblation(model_name)
        
    if run_all:
        run_all_layers(data_loader, model_name, save_dir, skip_exists)
    else:
        run_one(data_loader, model_name, save_dir, layer_idx, head_idx)
        
def run_one(data_loader, model_name, save_dir, layer_idx, head_idx):
    model = timmAblation(model_name)
    logging.info(f"On layer {layer_idx}, head {head_idx}")
    model.eval()
    model = model.cuda()

    # Ablate model
    model.ablate_attn_head(layer_idx, head_idx) 

    correct_predictions = 0
    total_predictions = 0

    # Determine the output size of your model
    output_size = model(torch.randn(1, 3, 224, 224).cuda()).shape[1]

    save_path = os.path.join(save_dir, f'layer{layer_idx}_head{head_idx}')
    os.makedirs(save_path, exist_ok=True)

    # Open an HDF5 file to save logits and labels
    with h5py.File(os.path.join(save_path, f'logits_and_labels.h5'), 'w') as h5f:
        # Create a dataset to store logits
        logits_dataset = h5f.create_dataset('logits', shape=(1, output_size), maxshape=(None, output_size), chunks=True)
        # Create a dataset to store labels
        labels_dataset = h5f.create_dataset('labels', shape=(1,), maxshape=(None,), dtype='i', chunks=True)
        logging.info("HDF5 file created for storing logits and labels.")

        processed_datapoints = 0
        for batch_num, (inputs, labels) in tqdm(enumerate(data_loader), total=len(data_loader)):

            with torch.no_grad():
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = model(inputs)

                probabilities = F.softmax(outputs, dim=1)
                predicted_labels = probabilities.argmax(dim=1)

                correct_predictions += (predicted_labels == labels).sum().item()
                total_predictions += labels.size(0)

                logits = outputs.cpu().numpy()
                labels = labels.cpu().numpy()

                batch_size = logits.shape[0]
                logits_dataset.resize((processed_datapoints + batch_size), axis=0)
                logits_dataset[processed_datapoints:] = logits
                labels_dataset.resize((processed_datapoints + batch_size), axis=0)
                labels_dataset[processed_datapoints:] = labels
                processed_datapoints += batch_size

    #         logging.info(f"Processed batch {batch_num + 1}/{len(data_loader)}, Total datapoints processed: {processed_datapoints}")

    total_accuracy = correct_predictions / total_predictions
    logging.info(f"All logits and labels saved successfully. Total accuracy: {total_accuracy:.4f}")
    with open(os.path.join(save_path, 'accuracy.txt'), 'w') as acc_file:
        acc_file.write(f"Accuracy for Layer {layer_idx}, Head {head_idx}: {total_accuracy:.4f}")
    del model
    
def run_all_layers(data_loader, model, save_dir, num_layers=12, num_heads=12, skip_exists=True):
    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            # Construct the directory path for the current layer and head index
            if attn:
            else:
                save_path = os.path.join(save_dir, f'layer{layer_idx}_h')

            # Check if the directory already exists
            if skip_exists and os.path.exists(save_path):
                # If skipping is enabled and the directory exists, log a message and skip this configuration
                logging.info(f"Skipping Layer {layer_idx}, Head {head_idx} as directory already exists.")
                continue
            
            run_one(data_loader, model_name, save_dir, layer_idx, head_idx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save logits of all ImageNet datapoints for future analysis.")
    parser.add_argument("--imagenet_path", type=str, default= '/network/datasets/imagenet.var/imagenet_torchvision/val/', help="Path to the ImageNet dataset.")
    parser.add_argument("--save_dir", type=str, default = '/network/scratch/s/sonia.joseph/imagenet_logits', help="Directory to save logits and labels.")
    parser.add_argument("--run_all", action='store_true', help="run all")
    parser.add_argument("--layer_idx", type=int, help="")
    parser.add_argument("--head_idx", type=int, help="")
    parser.add_argument("--skip_exists", action='store_true', help="Skip processing if directory already exists.", default=True)
    

    args = parser.parse_args()
    main(args.imagenet_path, args.save_dir, args.run_all, args.layer_idx, args.head_idx, args.skip_exists)
    
    #  imagenet_path = '/network/datasets/imagenet.var/imagenet_torchvision/val/'
    # save_dir = '/network/scratch/s/sonia.joseph/imagenet_logits'