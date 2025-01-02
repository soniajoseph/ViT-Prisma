import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
from typing import Optional, Dict
from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from typing import Optional, Dict
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from typing import Optional, Dict
from pathlib import Path

class ConceptualCaptionsLocalDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str,
        transform: Optional[transforms.Compose] = None,
        labels_file: str = "Image_Labels_Subset_Train_GCC-Labels-training.tsv"
    ):
        assert split in ['train', 'validation'], "Split must be 'train' or 'validation'"
        
        self.root_dir = Path(root_dir)
        self.split = split
        self.images_dir = self.root_dir / ('Train' if split == 'train' else 'Validation')
        
        # Read labels and ensure image_id is string
        labels_path = self.root_dir / labels_file
        self.labels_df = pd.read_csv(
            labels_path, 
            sep='\t',
            names=['image_id', 'caption'],
            dtype={'image_id': str}  # Ensure image_id is string
        )
        
        # Create image paths directly from image IDs
        self.image_files = [
            self.images_dir / str(img_id)[:3] / f"{img_id}-0.jpg"
            for img_id in self.labels_df['image_id']
        ]
        self.image_ids = self.labels_df['image_id'].tolist()
        self.id_to_caption = dict(zip(self.labels_df['image_id'], self.labels_df['caption']))
        
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path = self.image_files[idx]
        image_id = self.image_ids[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
            caption = self.id_to_caption[image_id]
            
            return {
                'image': image,
                'caption': caption,
                'image_id': image_id,
                'valid': torch.tensor(1)
            }
            
        except Exception as e:
            return {
                'image': torch.zeros((3, 224, 224)),
                'caption': '',
                'image_id': image_id,
                'valid': torch.tensor(0)
            }

def get_cc_dataloaders(
    root_dir: str,
    batch_size: int = 32,
    num_workers: int = 4
) -> tuple[DataLoader, DataLoader]:
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = ConceptualCaptionsLocalDataset(
        root_dir=root_dir,
        split='train',
        transform=transform
    )
    
    val_dataset = ConceptualCaptionsLocalDataset(
        root_dir=root_dir,
        split='validation',
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

if __name__ == "__main__":
    root_dir = "/network/datasets/conceptualcaptions"
    
    try:
        train_loader, val_loader = get_cc_dataloaders(
            root_dir=root_dir,
            batch_size=32,
            num_workers=4
        )
        
        print("\nTesting data loading...")
        for i, batch in enumerate(train_loader):
            if i >= 2:
                break
            print(f"\nBatch {i}:")
            print(f"Images shape: {batch['image'].shape}")
            print(f"Valid samples: {batch['valid'].sum().item()}/{len(batch['valid'])}")
            print(f"Sample caption: {batch['caption'][0]}")
            print(f"Sample image ID: {batch['image_id'][0]}")
            
    except Exception as e:
        print(f"\nError: {str(e)}")