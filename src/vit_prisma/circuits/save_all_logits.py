import timm
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import h5py
import numpy as np
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO)

# 1. Load and preprocess the ImageNet dataset
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

imagenet_data = datasets.ImageFolder('path_to_imagenet/train', transform=data_transforms)
data_loader = DataLoader(imagenet_data, batch_size=64, shuffle=True)
logging.info(f"ImageNet dataset loaded and preprocessed. Total datapoints: {len(imagenet_data)}")

# 2. Load a pre-trained model from timm
model = timm.create_model('vit_base_patch32_224', pretrained=True)
model.eval()
logging.info("Model loaded: vit_base_patch32_224")

# If you're using a GPU
if torch.cuda.is_available():
    model = model.cuda()
    logging.info("Using GPU for inference.")
    
correct_predictions = 0
total_predictions = 0

# Open an HDF5 file to save logits
with h5py.File('logits.h5', 'w') as h5f:
    # Create a dataset to store logits
    dset = h5f.create_dataset('logits', shape=(1, 1000), maxshape=(None, 1000), chunks=True)
    logging.info("HDF5 file created for storing logits.")

    processed_datapoints = 0
    for batch_num, (inputs, labels) in enumerate(data_loader):
        if torch.cuda.is_available():
            inputs = inputs.cuda()

        with torch.no_grad():
            outputs = model(inputs)
            
            probabilities = F.softmax(outputs, dim=1)
            predicted_labels = probabilities.argmax(dim=1)

            # Accumulate the number of correct predictions
            correct_predictions += (predicted_labels == labels).sum().item()
            total_predictions += labels.size(0)
            
            logits = outputs.cpu().numpy()
            
            # Resize HDF5 dataset to accommodate new logits
            batch_size = logits.shape[0]
            dset.resize((processed_datapoints + batch_size), axis=0)
            dset[processed_datapoints:] = logits
            processed_datapoints += batch_size

        logging.info(f"Processed batch {batch_num + 1}/{len(data_loader)}, Total datapoints processed: {processed_datapoints}")

total_accuracy = correct_predictions / total_predictions
logging.info(f"All logits saved successfully. Total accuracy: {total_accuracy:.4f}")
