from typing import Tuple

import torch
from torch.utils.data import random_split, Dataset
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import RandAugment


def get_cifar_transforms(augmentation: bool, image_size: int = 128, visualisation=False):
    cifar_transforms = [
        transforms.ToTensor(),
    ]

    if visualisation:
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
        return IndexPreservingSubset(cifar10_train_dataset.dataset,
                                     indices=cifar10_train_dataset.indices), IndexPreservingSubset(
            cifar10_val_dataset.dataset, indices=cifar10_val_dataset.indices), cifar10_test_dataset
    else:
        return cifar10_train_dataset, cifar10_val_dataset, cifar10_test_dataset