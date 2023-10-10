import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os

class DSpritesDataset(Dataset):
    def __init__(self, data_path):
        # Load the dataset
        data = np.load(data_path, allow_pickle=True, encoding='latin1')
        self.images = data['imgs']
        self.latents_values = data['latents_values']
        self.latents_classes = data['latents_classes']

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Convert image from numpy array to PyTorch tensor and add channel dimension
        image = torch.tensor(self.images[idx], dtype=torch.float32).unsqueeze(0)
        latents = torch.tensor(self.latents_values[idx], dtype=torch.float32)
        classes = torch.tensor(self.latents_classes[idx], dtype=torch.int64)
        return image, latents, classes
    

data_path = '/network/scratch/s/sonia.joseph/datasets/dsprites.npz'

dsprites_dataset = DSpritesDataset(data_path)
dataloader = DataLoader(dsprites_dataset, batch_size=32, shuffle=True)

for images, latents_values, latents_classes in dataloader:
    # Your training or evaluation code here
    print(images.shape)
    print(latents_values)
    print(latents_classes)

    break