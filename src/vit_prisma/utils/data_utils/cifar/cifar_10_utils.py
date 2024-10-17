import glob
import json
import os
from typing import Tuple

import numpy as np
import torch
import torchvision
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split, Dataset
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import RandAugment

from vit_prisma.dataloaders.imagenet_dataset import (
    ImageNetValidationDataset,
)
from vit_prisma.utils.constants import DATA_DIR
from vit_prisma.utils.data_utils.imagenet.imagenet_utils import setup_imagenet_paths
from vit_prisma.utils.data_utils.loader_utils import TransformedSubset


# class Cifar10DatasetWithIndex(CIFAR10):
#
#     def __getitem__(self, index: int):
#         img, target = super().__getitem__(index)
#         return img, target, index

# class DatasetWithIndex(torch.utils.data.Subset):
#     def __getitem__(self, current_index: int):
#         return *super().__getitem__(current_index), current_index

class IndexPreservingSubset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        # Return the requested index 'idx' along with the data
        # Instead of dataset[indices[idx]]
        return *self.dataset[self.indices[idx]], idx

    def __len__(self):
        return len(self.indices)


def get_cifar_transforms(augmentation: bool, image_size: int = 128, visualisation=False):
    cifar_transforms = [
        transforms.ToTensor(),
    ]

    if visualisation:  # TODO EdS: Is this the wrong way around
        cifar_transforms = cifar_transforms + [transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),]

    cifar_transforms = cifar_transforms + [transforms.Resize((image_size, image_size)),]

    if augmentation:
        cifar_transforms = [
            transforms.RandomResizedCrop(128, scale=(0.8, 1.0), ratio=(0.9, 1.1)),  # Moderate crop
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Added color jitter
            RandAugment(2, 10),
        ] + cifar_transforms

    transform = transforms.Compose(cifar_transforms)

    return transform


def get_mnist_transforms(augmentation: bool, image_size: int = 128, visualisation=False):
    cifar_transforms = [
        transforms.ToTensor(),
    ]

    if not visualisation:
        cifar_transforms = cifar_transforms + [transforms.Normalize((0.1307), (0.3081)),]

    cifar_transforms = cifar_transforms + [transforms.Resize((image_size, image_size)),]

    if augmentation:
        cifar_transforms = [
            transforms.RandomResizedCrop(128, scale=(0.8, 1.0), ratio=(0.9, 1.1)),  # Moderate crop
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Added color jitter
            RandAugment(2, 10),
        ] + cifar_transforms

    transform = transforms.Compose(cifar_transforms)

    return transform


def get_imagenet_transforms(augmentation: bool, image_size: int = 128, visualisation=False):
    imagenet_transforms = [
        transforms.ToTensor(),
    ]

    if not visualisation:
        imagenet_transforms = imagenet_transforms + [transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),]

    imagenet_transforms = imagenet_transforms + [transforms.Resize((image_size, image_size)),]

    if augmentation:
        imagenet_transforms = [
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),  # Moderate crop
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Added color jitter
            RandAugment(2, 10),
        ] + imagenet_transforms

    transform = transforms.Compose(imagenet_transforms)

    return transform


def load_cifar_10(
    dataset_path: str,
    split_size: float = 0.8,
    augmentation: bool = False,
    image_size: int = 128,
    with_index=False,
    visualisation=False,
) -> Tuple[Dataset, Dataset, Dataset]:
    """Load CIFAR-10 from Torchvision. It will cache the data locally."""

    if visualisation:
        train_transform = get_cifar_transforms(augmentation, image_size, visualisation=True)
        test_transform = get_cifar_transforms(augmentation=False, image_size=image_size, visualisation=True)
    else:
        train_transform = get_cifar_transforms(augmentation, image_size, visualisation=False)
        test_transform = get_cifar_transforms(augmentation=False, image_size=image_size, visualisation=False)

    print(f"Downloading or fetching the CIFAR-10 dataset to: {dataset_path}")

    cifar10_trainset = datasets.CIFAR10(
        root=dataset_path,
        train=True,
        download=True,
        transform=train_transform,
    )
    cifar10_test_dataset = datasets.CIFAR10(
        root=dataset_path,
        train=False,
        download=True,
        transform=test_transform,
    )

    train_size = int(len(cifar10_trainset) * split_size)
    val_size = len(cifar10_trainset) - train_size

    cifar10_train_dataset, cifar10_val_dataset = random_split(
        cifar10_trainset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"There are {len(cifar10_train_dataset)} samples in the train dataset")
    print(f"There are {len(cifar10_val_dataset)} samples in the validation dataset")
    print(f"There are {len(cifar10_test_dataset)} samples in the test dataset")
    # Subset(cifar10_test_dataset, indices=list(range(len(cifar10_test_dataset))))
    if with_index:
        return IndexPreservingSubset(cifar10_train_dataset.dataset, indices=cifar10_train_dataset.indices), IndexPreservingSubset(cifar10_val_dataset.dataset, indices=cifar10_val_dataset.indices), cifar10_test_dataset
    else:
        # return cifar10_train_dataset, cifar10_val_dataset, Dataset(cifar10_test_dataset
        return cifar10_train_dataset, cifar10_val_dataset, cifar10_test_dataset


def load_mnist(
    dataset_path: str,
    split_size: float = 0.8,
    augmentation: bool = False,
    image_size: int = 128,
    with_index=False,
    visualisation=False,
) -> Tuple[Dataset, Dataset, Dataset]:
    """Load MNIST from Torchvision. It will cache the data locally."""

    if visualisation:
        train_transform = get_mnist_transforms(augmentation, image_size, visualisation=False)
        test_transform = get_mnist_transforms(augmentation=False, image_size=image_size, visualisation=True)
    else:
        train_transform = get_mnist_transforms(augmentation, image_size, visualisation=False)
        test_transform = get_mnist_transforms(augmentation=False, image_size=image_size, visualisation=False)

    mnist_trainset = datasets.MNIST(
        root=dataset_path,
        train=True,
        download=True,
        transform=train_transform,
    )
    mnist_test_dataset = datasets.MNIST(
        root=dataset_path,
        train=False,
        download=True,
        transform=test_transform,
    )

    train_size = int(len(mnist_trainset) * split_size)
    val_size = len(mnist_trainset) - train_size

    mnist_train_dataset, mnist_val_dataset = random_split(
        mnist_trainset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"There are {len(mnist_train_dataset)} samples in the train dataset")
    print(f"There are {len(mnist_val_dataset)} samples in the validation dataset")
    print(f"There are {len(mnist_test_dataset)} samples in the test dataset")
    # Subset(cifar10_test_dataset, indices=list(range(len(cifar10_test_dataset))))
    if with_index:
        return IndexPreservingSubset(
            mnist_train_dataset.dataset, indices=mnist_train_dataset.indices
        ), IndexPreservingSubset(
            mnist_val_dataset.dataset,
            indices=mnist_val_dataset.indices
        ), mnist_test_dataset
    else:
        return mnist_train_dataset, mnist_val_dataset, mnist_test_dataset

def load_cifar_100(
    dataset_path: str,
    split_size: float = 0.8,
    augmentation: bool = False,
    image_size: int = 128,
) -> Tuple[Dataset, Dataset, Dataset]:
    """Load CIFAR-100 from Torchvision. It will cache the data locally."""

    train_transform, test_transform = get_cifar_transforms(augmentation, image_size)

    cifar100_trainset = datasets.CIFAR100(
        root=dataset_path,
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )
    cifar100_test_dataset = datasets.CIFAR100(
        root=dataset_path,
        train=False,
        download=True,
        transform=test_transform,
    )

    train_idx, val_idx = train_test_split(
        range(len(cifar100_trainset)),
        test_size=0.2,
        random_state=42
    )

    cifar10_train_dataset = TransformedSubset(cifar100_trainset, train_idx, train_transform)
    cifar10_val_dataset = TransformedSubset(cifar100_trainset, val_idx, test_transform)

    print(f"There are {len(cifar10_train_dataset)} samples in the train dataset")
    print(f"There are {len(cifar10_val_dataset)} samples in the validation dataset")
    print(f"There are {len(cifar100_test_dataset)} samples in the test dataset")

    return cifar10_train_dataset, cifar10_val_dataset, cifar100_test_dataset


class TinyImageNetDataset(Dataset):
    def __init__(self, root, split='train', transform=None):
        """
        Args:
            root (str): Path to tiny-imagenet-200 directory
            split (str): 'train', 'val', or 'test'
            transform: Optional transforms to apply
        """
        self.transform = transform
        self.split = split
        self.root = root
        self.samples = []

        # Load class to idx mapping
        self.class_to_idx = {}
        wnids_path = os.path.join(root, 'wnids.txt')
        if not os.path.exists(wnids_path):
            raise FileNotFoundError(f"Could not find wnids.txt at {wnids_path}")

        with open(wnids_path, 'r') as f:
            for idx, line in enumerate(f.readlines()):
                self.class_to_idx[line.strip()] = idx

        if split == 'train':
            train_dir = os.path.join(root, 'train')
            if not os.path.exists(train_dir):
                raise FileNotFoundError(f"Could not find train directory at {train_dir}")

            # For training, use class directories
            class_dirs = [d for d in os.listdir(train_dir)
                          if os.path.isdir(os.path.join(train_dir, d))]

            for class_dir in class_dirs:
                class_idx = self.class_to_idx[class_dir]
                images_dir = os.path.join(train_dir, class_dir, 'images')
                if not os.path.exists(images_dir):
                    continue

                image_paths = glob.glob(os.path.join(images_dir, '*.JPEG'))
                self.samples.extend([(path, class_idx) for path in image_paths])

        elif split == 'val':
            val_dir = os.path.join(root, 'val')
            annotations_file = os.path.join(val_dir, 'val_annotations.txt')

            if not os.path.exists(annotations_file):
                raise FileNotFoundError(f"Could not find val_annotations.txt at {annotations_file}")

            # For validation, use val_annotations.txt
            with open(annotations_file, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:  # Ensure we have both image name and class id
                        image_name, class_id = parts[0], parts[1]
                        image_path = os.path.join(val_dir, 'images', image_name)
                        if os.path.exists(image_path) and class_id in self.class_to_idx:
                            self.samples.append((image_path, self.class_to_idx[class_id]))

        elif split == 'test':
            test_dir = os.path.join(root, 'test', 'images')
            if not os.path.exists(test_dir):
                raise FileNotFoundError(f"Could not find test directory at {test_dir}")

            # For test, we don't have labels (just return -1 as label)
            image_paths = glob.glob(os.path.join(test_dir, '*.JPEG'))
            self.samples.extend([(path, -1) for path in image_paths])

        else:
            raise ValueError(f"Split {split} not recognized. Use 'train', 'val', or 'test'")

        if len(self.samples) == 0:
            raise RuntimeError(f"Found 0 files in {split} split at {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            with Image.open(path) as img:
                image = img.convert('RGB')
        except Exception as e:
            print(f"Error loading image {path}: {str(e)}")
            # Return a black image and the label if image loading fails
            image = Image.new('RGB', (64, 64))  # TinyImageNet size

        if self.transform:
            image = self.transform(image)
        return image, label


class MixupLabelSmoothingDataLoader:
    def __init__(self, dataloader, alpha=0.2, mixup_prob=0.5, smoothing=0.1):
        self.dataloader = dataloader
        self.alpha = alpha
        self.mixup_prob = mixup_prob
        self.smoothing = smoothing
        self.n_classes = 200  # TinyImageNet has 200 classes

    def smooth_labels(self, labels):
        smooth_labels = torch.zeros(labels.size(0), self.n_classes, device=labels.device)
        smooth_labels.fill_(self.smoothing / (self.n_classes - 1))
        smooth_labels.scatter_(1, labels.unsqueeze(1), 1.0 - self.smoothing)
        return smooth_labels

    def __iter__(self):
        for batch in self.dataloader:
            images, labels = batch

            # Convert labels to one-hot with smoothing
            labels = self.smooth_labels(labels)

            if np.random.random() < self.mixup_prob:
                lam = np.random.beta(self.alpha, self.alpha)
                idx = torch.randperm(images.size(0))
                mixed_images = lam * images + (1 - lam) * images[idx]
                mixed_labels = lam * labels + (1 - lam) * labels[idx]
                yield mixed_images, mixed_labels
            else:
                yield images, labels

    def __len__(self):
        return len(self.dataloader)

def load_tinyimagenet(
    batch_size: int = 32,
) -> Tuple[Dataset, Dataset, Dataset]:

    tinyimagenet_train_transforms = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.RandomResizedCrop(128, scale=(0.8, 1.0), ratio=(0.9, 1.1)),  # Moderate crop
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Added color jitter
            RandAugment(2, 10),
            transforms.ToTensor(),
            transforms.Normalize((0.480237, 0.448067, 0.397504), (0.276437, 0.268864, 0.281590)),
        ]
    )

    tinyimagenet_transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.480237, 0.448067, 0.397504), (0.276437, 0.268864, 0.281590)),
    ])

    train_dataset = TinyImageNetDataset(DATA_DIR / "tinyimagenet/tiny-imagenet-200", split='train', transform=tinyimagenet_train_transforms)
    val_dataset = TinyImageNetDataset(DATA_DIR / "tinyimagenet/tiny-imagenet-200", split='val', transform=tinyimagenet_transforms)
    test_dataset = TinyImageNetDataset(DATA_DIR / 'tinyimagenet/tiny-imagenet-200', split="test", transform=tinyimagenet_transforms)

    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"There are {len(train_dataset)} samples in the train dataset")
    print(f"There are {len(val_dataset)} samples in the validation dataset")
    print(f"There are {len(test_dataset)} samples in the test dataset")

    return train_dataset, val_dataset, test_dataset


class ImageNet100Dataset(Dataset):
    def __init__(self, root, split='train', transform=None):
        """
        Args:
            root (str): Path to ImageNet-100 directory
            split (str): 'train' or 'val'
            transform: Optional transforms to apply
        """
        self.transform = transform
        self.split = split
        self.root = root
        self.samples = []

        # Load class mapping
        class_mapping_file = os.path.join(root, 'Labels.json')
        if not os.path.exists(class_mapping_file):
            raise FileNotFoundError(f"Could not find Labels.json at {class_mapping_file}")

        with open(class_mapping_file, 'r') as f:
            self.class_mapping = json.load(f)

        # Create index mapping
        self.class_to_idx = {class_id: idx for idx, class_id in enumerate(sorted(self.class_mapping.keys()))}

        # Determine which directories to process
        if split == 'train':
            split_dirs = ['train.X1', 'train.X2', 'train.X3', 'train.X4']
        elif split == 'val':
            split_dirs = ['val.X']
        else:
            raise ValueError(f"Split {split} not recognized. Use 'train' or 'val'")

        # Process each split directory
        for split_dir in split_dirs:
            split_path = os.path.join(root, split_dir)
            if not os.path.exists(split_path):
                raise FileNotFoundError(f"Could not find directory at {split_path}")

            # Process each class directory
            for class_id in self.class_mapping.keys():
                class_path = os.path.join(split_path, class_id)
                if not os.path.exists(class_path):
                    continue

                # Get all JPEG images in the class directory
                image_paths = glob.glob(os.path.join(class_path, '*.JPEG'))
                class_idx = self.class_to_idx[class_id]

                for img_path in image_paths:
                    self.samples.append((img_path, class_idx))

        if len(self.samples) == 0:
            raise RuntimeError(f"Found 0 files in {split} split at {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            with Image.open(path) as img:
                image = img.convert('RGB')
        except Exception as e:
            print(f"Error loading image {path}: {str(e)}")
            # Return a black image and the label if image loading fails
            image = Image.new('RGB', (224, 224))

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_num_classes(self):
        """Returns number of classes"""
        return len(self.class_mapping)

    def get_class_name(self, idx):
        """Returns class name for given index"""
        for class_id, class_idx in self.class_to_idx.items():
            if class_idx == idx:
                return self.class_mapping[class_id]
        return None

    @staticmethod
    def get_normalization_values():
        """Returns standard ImageNet normalization values"""
        return {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        }


def load_imagenet100(img_size=224) -> Tuple[Dataset, Dataset]:
    tinyimagenet_train_transforms = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),  # Moderate crop
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Added color jitter
            RandAugment(2, 10),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    tinyimagenet_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    train_dataset = ImageNet100Dataset(DATA_DIR / "imagenet100", split='train', transform=tinyimagenet_train_transforms)
    val_dataset = ImageNet100Dataset(DATA_DIR / "imagenet100", split='val', transform=tinyimagenet_transforms)

    print(f"There are {len(train_dataset)} samples in the train dataset")
    print(f"There are {len(val_dataset)} samples in the validation dataset")
    print(f"The images are of size: {img_size}")
    return train_dataset, val_dataset


def load_imagenet(
    dataset_path: str,
    dataset_train_path: str,
    dataset_val_path: str,
    augmentation: bool = False,
    image_size: int = 128,
    with_index=False,
    visualisation=False,
) -> Tuple[Dataset, Dataset, Dataset]:
    if visualisation:
        train_transform = get_imagenet_transforms(augmentation, image_size, visualisation=True)
        test_transform = get_imagenet_transforms(augmentation=False, image_size=image_size, visualisation=True)
    else:
        train_transform = get_imagenet_transforms(augmentation, image_size, visualisation=False)
        test_transform = get_imagenet_transforms(augmentation=False, image_size=image_size, visualisation=False)

    print(f"Downloading or fetching the Imagenet1k dataset to: {dataset_path}")

    imagenet_paths = setup_imagenet_paths(dataset_path)
    train_data = torchvision.datasets.ImageFolder(
        dataset_train_path, transform=train_transform
    )
    val_data = ImageNetValidationDataset(
        dataset_val_path,
        imagenet_paths["label_strings"],
        imagenet_paths["val_labels"],
        test_transform,
        return_index=with_index,
    )
    if visualisation:
        val_data_visualize = ImageNetValidationDataset(
            dataset_val_path,
            imagenet_paths["label_strings"],
            imagenet_paths["val_labels"],
            torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize((image_size, image_size)),
                    torchvision.transforms.ToTensor(),
                ]
            ),
            return_index=with_index,
        )
    else:
        val_data_visualize = None

    print(f"There are {len(train_data)} samples in the train dataset")
    print(f"There are {len(val_data)} samples in the validation dataset")

    return train_data, val_data, val_data_visualize
