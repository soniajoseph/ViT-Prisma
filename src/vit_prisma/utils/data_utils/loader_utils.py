import random

from torch.utils.data import Dataset


class SubsetDataset(Dataset):
    """For testing purposes - reduce a Dataset to N of its samples."""

    def __init__(self, dataset, n, random_subset=False):
        self.dataset = dataset
        self.n = min(n, len(dataset))
        if random_subset:
            self.indices = random.sample(range(len(dataset)), self.n)
        else:
            self.indices = list(range(self.n))

        self.data = []
        for idx in self.indices:
            self.data.append(self.dataset[idx])

        del self.dataset

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.n


class TransformedSubset(Dataset):
    """Create custom datasets with different transforms"""

    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, idx):
        img, label = self.dataset[self.indices[idx]]
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.indices)
