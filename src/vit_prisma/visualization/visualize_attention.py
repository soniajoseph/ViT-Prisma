# Helper function: javascript visualization
import numpy as np
import json
from IPython.core.display import display, HTML
import string
import random

# Helper function to plot attention patterns, hide this function

import matplotlib.pyplot as plt
import numpy as np

def plot_attn_heads(total_activations, n_heads = 4, n_layers = 1, img_shape=32, idx=0, figsize = (20, 20),
    global_min_max=False, global_normalize=False, fourier_transform_local=False, fourier_transform_global=False, graph_type = "imshow_graph", cmap='viridis'):

    log_transform = False
    save_figure = False

    total_data = np.zeros((n_layers, n_heads, img_shape, img_shape))
    if global_min_max or global_normalize or fourier_transform_global:
        for i in range(n_layers):
            for j in range(n_heads):
                data = total_activations[i][idx,j,:,:]
                if log_transform:
                    data = log10_stable(data)
                if fourier_transform_global:
                    data = np.abs(np.fft.fft2(data))

                total_data[i,j,:,:] = data

        total_min = np.min(total_data)
        total_max = np.max(total_data)
        print(f"Total Min: {total_min}, Total Max: {total_max}")

        if global_normalize:
            total_data = -1 + 2 * (total_data - total_min) / (total_max - total_min)
            # put back in list
            total_activations = []
            for i in range(12):
                matrix = np.expand_dims(total_data[idx,:,:,:], axis=0)
                total_activations.append(matrix)

    fig, axes = plt.subplots(n_layers, n_heads, figsize=figsize)
    total_data_dict = {}

    is_1d_axes = len(axes.shape) == 1


    # Iterate over each row and column
    for i in range(n_layers): # Layer

        total_data_dict[f"Layer_{i}"] = {}

        for j in range(n_heads): # Head

            # Get Data
            data = total_activations[i][idx,j,:,:]

            # Plot the imshow plot in the corresponding subplot
            if graph_type == "histogram_graph":
                data = data.flatten()
                if log_transform:
                    axes[i, j].set_yscale('log')
                    axes[i, j].set_xscale('log')
    #             axes[i, j].set_title(f"Layer {layer_num}, Head {head_num} Attention")
                n, bins, patches = axes[i, j].hist(data, bins=100)

            elif graph_type == "imshow_graph": # Plot imshow

                if log_transform:
                    data = log10_stable(data)
                    data = np.round(data, 5)
                if fourier_transform_local:
                    data = np.abs(np.fft.fft2(data))

                if global_min_max or fourier_transform_global:
                    vmin = total_min
                    vmax = total_max
                elif global_normalize:
                    vmin = -1
                    vmax = 1
                else:
                    vmax = np.max(data)
                    vmin = np.min(data)
                
                if is_1d_axes:
                    im = axes[j].imshow(np.round(data,5), vmin=vmin, vmax=vmax, cmap=cmap)
                    axes[j].axis('off')
                else:
                    im = axes[i, j].imshow(np.round(data,5), vmin=vmin, vmax=vmax, cmap=cmap)
                    axes[i, j].axis('off')
                    
                total_data_dict[f"Layer_{i}"][f"Head_{j}"] = [[round(num, 5) for num in row] for row in data.tolist()]

                # Remove axis labels and ticks
            if i == 0:
                if is_1d_axes:
                    axes[j].set_title(f"Head {j}", fontsize=12, pad=5)
                else:
                    axes[i, j].set_title(f"Head {j}", fontsize=12, pad=5)
            if j == 0:
                if is_1d_axes:
                    axes[j].text(-0.3, 0.5, f"Layer {i}", fontsize=12, rotation=90, ha='center', va='center', transform=axes[j].transAxes)
                else:
                    axes[i, j].text(-0.3, 0.5, f"Layer {i}", fontsize=12, rotation=90, ha='center', va='center', transform=axes[i, j].transAxes)

    # Add colorbar
    if graph_type == "imshow_graph" and global_min_max:
        cbar_ax = fig.add_axes([0.15, 0.95, 0.7, 0.01])
        # Create a colorbar in the new axes
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal', shrink=.4)
        if global_min_max:
            cbar.set_label('Color Scale Across All Attention Heads', fontsize=16)
        else:
            cbar.set_label('Color Scale for Each Individual Attention Head', fontsize=16)

    # Adjust the spacing between subplots
    if graph_type == "histogram_graph":
        plt.subplots_adjust(wspace=0.6, hspace=0.6)
    elif graph_type == "imshow_graph":
        plt.subplots_adjust(wspace=0.1, hspace=0.2)

    if global_min_max:
        plt.suptitle(f'Attention Scores for Image Idx {idx}', fontsize=20, y=.98)
    else:
        plt.suptitle(f'Attention Scores for Image Idx {idx} (Colorbar per Image, not Global)', fontsize=20, y=.98)

    # plt.tight_layout()
    # save_path = './figures/attn_pattern_imshow_log_globalcolor.png'
    # save_path = './figures/attn_pattern_imshow_log.png'
    # if save_figure:
    #     save_path = './figures/attn_scores_truck_imshow_log_globalcolor.png'
    #     plt.savefig(save_path, bbox_inches='tight')
    #     print(f"Saved at {save_path}")

    plt.show()
