# Helper function: javascript visualization
import numpy as np
import json
from IPython.core.display import display, HTML
import string
import random

# Helper function to plot attention patterns, hide this function

import matplotlib.pyplot as plt
import numpy as np

def plot_attn_heads(total_activations, n_heads=12, n_layers=12, img_shape=50, idx=0, figsize=(20, 20),
                    global_min_max=False, global_normalize=False, fourier_transform_local=False, log_transform=False,
                    fourier_transform_global=False, graph_type="imshow_graph", cmap='viridis'):

    # New shape handling: total_activations is now expected to be of shape [n_layers*n_heads, img_shape, img_shape]
    total_data = np.zeros((n_layers*n_heads, img_shape, img_shape))

    # Adjusted processing for flattened layer-heads structure
    if global_min_max or global_normalize or fourier_transform_global:
        for i in range(n_layers*n_heads):
            data = total_activations[i, :, :]
            if log_transform:
                data = np.log10(np.maximum(data, 1e-6))  # log10_stable equivalent
            if fourier_transform_global:
                data = np.abs(np.fft.fftshift(np.fft.fft2(data)))
            total_data[i, :, :] = data

        total_min, total_max = np.min(total_data), np.max(total_data)
        print(f"Total Min: {total_min}, Total Max: {total_max}")

        if global_normalize:
            total_data = -1 + 2 * (total_data - total_min) / (total_max - total_min)

    fig, axes = plt.subplots(n_layers, n_heads, figsize=figsize, squeeze=False)  # Ensure axes is always 2D array
    total_data_dict = {}

    for i in range(n_layers):
        total_data_dict[f"Layer_{i}"] = {}
        for j in range(n_heads):
            # Adjust indexing for the flattened layer-head structure
            linear_idx = i * n_heads + j
            data = total_data[linear_idx, :, :]

            if graph_type == "histogram_graph":
                data = data.flatten()
                axes[i, j].hist(data, bins=100, log=log_transform, cmap=cmap)
            elif graph_type == "imshow_graph":
                if fourier_transform_local:
                    data = np.abs(np.fft.fftshift(np.fft.fft2(data)))
                vmin, vmax = (total_min, total_max) if (global_min_max or global_normalize) else (data.min(), data.max())
                im = axes[i, j].imshow(data, vmin=vmin, vmax=vmax, cmap=cmap)
                axes[i, j].axis('off')
                total_data_dict[f"Layer_{i}"][f"Head_{j}"] = data.tolist()

            axes[i, j].set_title(f"Head {j}", fontsize=12, pad=5) if i == 0 else None
            if j == 0:
                axes[i, j].text(-0.3, 0.5, f"Layer {i}", fontsize=12, rotation=90, ha='center', va='center', transform=axes[i, j].transAxes)

    # Add colorbar for imshow_graph
    if graph_type == "imshow_graph" and (global_min_max or global_normalize):
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        cbar_ax.set_title('Attention', size=12)

    plt.subplots_adjust(wspace=0.2, hspace=0.4)
    plt.suptitle(f'Attention for Image Idx {idx}', fontsize=20, y=0.93)
    plt.show()

