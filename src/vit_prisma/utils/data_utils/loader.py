from copy import deepcopy
from vit_prisma.utils.data_utils.cifar.cifar_10_utils import load_mnist
import torch
import torchvision
from torch.utils.data import Dataset, random_split, ConcatDataset

from vit_prisma.dataloaders.imagenet_dataset import (
    ImageNetValidationDataset,
)
from vit_prisma.transforms.open_clip_transforms import get_clip_val_transforms
from vit_prisma.utils.data_utils.cifar.cifar_10_utils import load_cifar_10, load_cifar_100
from vit_prisma.utils.data_utils.imagenet.imagenet_utils import setup_imagenet_paths


def load_dataset(cfg, model_type="clip", visualize=False):
    """Load and prepare datasets based on configuration and model type. Currently the
    function loads either ImageNet1K or CIFAR-10 datasets.
    """
    print(f"Dataset type: {cfg.dataset_name}") if cfg.verbose else None

    if cfg.dataset_name == "imagenet1k":
        if model_type == "clip":
            data_transforms = get_clip_val_transforms()
        else:
            raise ValueError("Invalid model type")

        imagenet_paths = setup_imagenet_paths(cfg.dataset_path)
        train_data = torchvision.datasets.ImageFolder(
            cfg.dataset_train_path, transform=data_transforms
        )
        val_data = ImageNetValidationDataset(
            cfg.dataset_val_path,
            imagenet_paths["label_strings"],
            imagenet_paths["val_labels"],
            data_transforms,
            return_index=True,
        )
        if visualize:
            val_data_visualize = ImageNetValidationDataset(
                cfg.dataset_val_path,
                imagenet_paths["label_strings"],
                imagenet_paths["val_labels"],
                torchvision.transforms.Compose(
                    [
                        torchvision.transforms.Resize((224, 224)),
                        torchvision.transforms.ToTensor(),
                    ]
                ),
                return_index=True,
            )
        else:
            val_data_visualize = None

        print(f"Train data length: {len(train_data)}") if cfg.verbose else None
        print(f"Validation data length: {len(val_data)}") if cfg.verbose else None
        return train_data, val_data, val_data_visualize
    elif cfg.dataset_name == "tinyimagenet":
        pass # TODO EdS:
    elif cfg.dataset_name == "cifar10":
        train_data, val_data, test_data = load_cifar_10(cfg.dataset_path)
        combined_dataset = ConcatDataset([train_data, test_data])
        print(f"Train data length: {len(combined_dataset)}") if cfg.verbose else None
        print(f"Validation data length: {len(val_data)}") if cfg.verbose else None
        return combined_dataset, val_data, None
    elif cfg.dataset_name == "cifar100":
        train_data, val_data, test_data = load_cifar_100(cfg.dataset_path)
        return test_data, val_data, val_data
    elif cfg.dataset_name == "mnist":
        train_data, val_data, test_data = load_mnist(cfg.dataset_path, augmentation=True, visualisation=False)
        return test_data, val_data, deepcopy(val_data)  # TODO EdS: The visualisation data should be unnormalised
    else:
        # raise error
        raise ValueError("Invalid dataset name")


def load_test_data(cfg):
    print(f"Dataset type: {cfg.dataset_name}") if cfg.verbose else None

    if cfg.dataset_name == "imagenet1k":
        if model_type == "clip":
            data_transforms = get_clip_val_transforms()
        else:
            raise ValueError("Invalid model type")

        imagenet_paths = setup_imagenet_paths(cfg.dataset_path)
        test_data = torchvision.datasets.ImageFolder(
            cfg.dataset_test_path, transform=data_transforms
        )
        # data = ImageNetValidationDataset(
        #     cfg.dataset_val_path,
        #     imagenet_paths["label_strings"],
        #     imagenet_paths["val_labels"],
        #     data_transforms,
        #     return_index=True,
        # )
        # data_visualize = ImageNetValidationDataset(
        #     cfg.dataset_val_path,
        #     imagenet_paths["label_strings"],
        #     imagenet_paths["val_labels"],
        #     torchvision.transforms.Compose(
        #         [
        #             torchvision.transforms.Resize((224, 224)),
        #             torchvision.transforms.ToTensor(),
        #         ]
        #     ),
        #     return_index=True,
        # )

        print(f"Test data length: {len(train_data)}") if cfg.verbose else None
        # print(f"Visualisation Test data length: {len(val_data)}") if cfg.verbose else None
        # return test_data, test_data_visualize
        return test_data
    elif cfg.dataset_name == "tinyimagenet":
        pass  # TODO EdS:
    elif cfg.dataset_name == "cifar10":
        train_data, val_data, test_data = load_cifar_10(cfg.dataset_path)
        combined_dataset = ConcatDataset([train_data, test_data])
        print(f"Train data length: {len(combined_dataset)}") if cfg.verbose else None
        print(f"Validation data length: {len(val_data)}") if cfg.verbose else None
        return combined_dataset, val_data, None
    elif cfg.dataset_name == "cifar100":
        train_data, val_data, test_data = load_cifar_100(cfg.dataset_path)
        return test_data, val_data, val_data
    else:
        # raise error
        raise ValueError("Invalid dataset name")

