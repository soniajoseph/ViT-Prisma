import torch
from torch.utils.data import DataLoader


def check_dataset_stats(dataset, num_samples=10000):
    # Convert dataset to DataLoader if it isn't already
    loader = DataLoader(dataset, batch_size=32, shuffle=True) if not isinstance(dataset, DataLoader) else dataset

    pixels = []
    means = []
    stds = []
    count = 0

    print("Checking first few images...")
    for images, _ in loader:
        # Check the data type and range of first batch
        if count == 0:
            print(f"Data type: {images.dtype}")
            print(f"Value range: [{images.min():.3f}, {images.max():.3f}]")

        # Collect statistics
        batch_mean = images.mean(dim=[0, 2, 3])
        batch_std = images.std(dim=[0, 2, 3])

        means.append(batch_mean)
        stds.append(batch_std)

        count += images.size(0)
        if count >= num_samples:
            break

    # Calculate overall statistics
    mean = torch.stack(means).mean(dim=0)
    std = torch.stack(stds).mean(dim=0)

    print("\nDataset Statistics:")
    print(f"Means per channel: {mean.tolist()}")
    print(f"Stds per channel: {std.tolist()}")

    # Interpretation
    if images.dtype == torch.uint8:
        print("\nInterpretation: Dataset is in raw format (needs ToTensor)")
    elif 0 <= images.min() and images.max() <= 1.0:
        print("\nInterpretation: Dataset has ToTensor applied but needs normalization")
    elif abs(mean.mean()) < 0.1 and 0.9 < std.mean() < 1.1:
        print("\nInterpretation: Dataset appears properly normalized")
    else:
        print("\nInterpretation: Dataset has some form of normalization/scaling - check your transforms")