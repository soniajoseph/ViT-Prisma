import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torchvision.transforms import transforms

def get_inverse(dataset="cifar"):
    if dataset == "cifar":
        return transforms.Normalize(
            mean=[-0.4914 / 0.2023, -0.4822 / 0.1994, -0.4465 / 0.2010],
            std=[1 / 0.2023, 1 / 0.1994, 1 / 0.2010],
        )
    elif dataset == "mnist":
        return transforms.Normalize(
            mean=[0.1307],
            std=[0.3081],
        )


def plot_image(image, unstandardise=True, dataset="cifar", save_path=None):
    plt.figure()
    plt.axis("off")

    if unstandardise:
        image = get_inverse(dataset)(image)

    if save_path:
        print(f"Saving image at: {save_path}")
        torchvision.utils.save_image(image, save_path)

    image = image.permute(1, 2, 0)
    plt.imshow(image, vmin=-2.5, vmax=2.0)


def display_grid_on_image(image, patch_size=32):
    image = inverse_transform(image)

    if isinstance(image, torch.Tensor):
        image = image.detach().numpy().transpose(1, 2, 0)

    if image.shape[0] != 128:
        image = image.transpose(1, 2, 0)

    # if image.max() <= 3.0:
    #     image = (image * 255).astype(np.uint8)

    num_patches = (image.shape[0] / patch_size) ** 2

    grid_size = int(np.sqrt(num_patches))

    # Calculate patch size
    patch_height = image.shape[0] // grid_size
    patch_width = image.shape[1] // grid_size

    # Overlay grida
    grid_image = np.copy(image)
    for i in range(1, grid_size):
        # Vertical lines
        grid_image[:, patch_width * i, :] = [255, 255, 255]
        # Horizontal lines
        grid_image[patch_height * i, :, :] = [255, 255, 255]

    plt.figure(figsize=(4, 4), dpi=100)  # Adjust figsize and dpi as needed

    # Place labels
    for i in range(grid_size):
        for j in range(grid_size):
            x_center = (j * patch_width) + (patch_width // 2)
            y_center = (i * patch_height) + (patch_height // 2)
            # Convert the patch index to 2D coordinates (row, column)
            patch_index = i * grid_size + j
            row, col = divmod(patch_index, grid_size)
            plt.text(
                x_center,
                y_center,
                f"{patch_index+1}",
                color="red",
                fontsize=8,
                ha="center",
                va="center",
            )

    # Display image with grid and labels
    # plt.figure(figsize=(8, 8))
    # plt.imshow(grid_image)
    plt.imshow(grid_image, vmin=-2.5, vmax=2.0)
    plt.axis("off")
    plt.show()
