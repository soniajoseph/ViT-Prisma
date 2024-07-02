from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure


def display_grid_on_image(image: Union[np.ndarray, torch.Tensor], patch_size: int = 32, return_plot: bool = False) -> Optional[matplotlib.figure.Figure]:
    """
    Separates an image into a grid of patches and overlays the grid on the image.

    Args:
        image (torch.Tensor): The input image, either as a numpy array or a PyTorch tensor. 
                              Dimensions of (H, W, C) if numpy or (C, H, W) if tensor
        patch_size (int, optional): The size of each patch in the grid. Default is 32.
        return_plot (bool, optional): If True, the function will return the plot figure. If False, it will display the plot. Default is False.

    Returns:
        matplotlib.figure.Figure or None: If return_plot is True, returns the matplotlib figure object. Otherwise, displays the image with the grid overlay.
    """

    if isinstance(image, torch.Tensor):
        image = image.detach().numpy().transpose(1, 2, 0)
    if image.shape[0] != 224:
        image = image.transpose(1, 2, 0)
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)

    num_patches = (image.shape[0] / patch_size) ** 2
    grid_size = int(np.sqrt(num_patches))

    # Calculate patch size
    patch_height = image.shape[0] // grid_size
    patch_width = image.shape[1] // grid_size

    # Overlay grid
    grid_image = np.copy(image)
    for i in range(1, grid_size):
        # Vertical lines
        grid_image[:, patch_width * i, :] = [255, 255, 255]
        # Horizontal lines
        grid_image[patch_height * i, :, :] = [255, 255, 255]

    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)  # Adjust figsize and dpi as needed

    # Place labels
    for i in range(grid_size):
        for j in range(grid_size):
            x_center = (j * patch_width) + (patch_width // 2)
            y_center = (i * patch_height) + (patch_height // 2)
            # Convert the patch index to 2D coordinates (row, column)
            patch_index = i * grid_size + j
            row, col = divmod(patch_index, grid_size)
            ax.text(x_center, y_center, f"{patch_index+1}", color='red', fontsize=8, ha='center', va='center')

    # Display image with grid and labels
    ax.imshow(grid_image)
    ax.axis('off')

    if return_plot:
        return fig
    else:
        plt.show()
