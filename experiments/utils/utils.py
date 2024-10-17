from collections import defaultdict
from itertools import islice

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import torch
from fancy_einsum import einsum
from torch.utils.data import random_split
from torchvision import datasets


def average_logit_value_across_all_classes(
    model,
    residual_stack,
    cache,
    mean=True,
):
    """We can project every intermediate CLS token output from each layer onto the
    residual directions. This is the equivalent of directly feeding the CLS output of each layer into the 10-way
    classification head to get a logit value for each class.
    """
    scaled_residual_stack = cache.apply_ln_to_stack(
        residual_stack, layer=-1, pos_slice=0
    )

    all_residual_directions = model.tokens_to_residual_directions(
        np.arange(10)
    )  # Get all residual directions
    logit_predictions = einsum(
        "layer batch d_model, batch d_model -> batch layer",
        scaled_residual_stack,
        all_residual_directions,
    )
    if mean:
        logit_predictions = logit_predictions.mean(axis=0)

    return logit_predictions


def residual_stack_to_logit(residual_stack, cache, answer_residual_directions):
    scaled_residual_stack = cache.apply_ln_to_stack(
        residual_stack, layer=-1, pos_slice=0
    )
    logit_predictions = einsum(
        "layer batch d_model, d_model -> layer",
        scaled_residual_stack,
        answer_residual_directions,
    )
    return logit_predictions


def plot_logit_boxplot(average_logits, labels):
    hovertexts = np.array([[CIFAR10_INDICES[i] for _ in range(17)] for i in range(10)])

    fig = go.Figure()
    data = []

    # if tensor, turn to numpy
    if isinstance(average_logits, torch.Tensor):
        average_logits = average_logits.detach().cpu().numpy()

    for i in range(average_logits.shape[1]):  # For each layer
        layer_logits = average_logits[:, i]
        hovertext = hovertexts[:, i]
        box = fig.add_trace(
            go.Box(
                y=layer_logits,
                name=f"{labels[i]}",
                text=hovertext,
                hoverinfo="y+text",
                boxpoints="all",
            )
        )
        data.append(box)

    fig.show()


def get_patch_logit_dictionary(patch_logit_directions, batch_idx=0, rank_label=None):
    patch_dictionary = defaultdict(list)
    # if tuple, get first entry
    if isinstance(patch_logit_directions, tuple):
        patch_logit_directions = patch_logit_directions[0]
    # Go through laeyrs of one batch
    for patch_idx, patches in enumerate(patch_logit_directions[batch_idx]):
        # Go through every patch and get max prediction
        for logits in patches:
            probs = torch.softmax(logits, dim=-1)
            # Get index of max prediction
            predicted_idx = int(torch.argmax(probs))
            logit = logits[predicted_idx].item()
            predicted_class_name = CIFAR10_INDICES[predicted_idx]
            if rank_label:
                # Where is the rank_label in the sorted list?
                rank_index = CIFAR10_LABELS[rank_label]
                sorted_list = torch.argsort(probs, descending=True)
                rank = np.where(sorted_list == rank_index)[0][0]
                patch_dictionary[patch_idx].append(
                    (logit, predicted_class_name, predicted_idx, rank)
                )
            else:
                patch_dictionary[patch_idx].append(
                    (logit, predicted_class_name, predicted_idx)
                )
    return patch_dictionary


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


def display_grid_on_image_with_heatmap(
    image,
    patch_dictionary,
    patch_size=32,
    layer_idx=-1,
    imagenet_class_to_emoji=CIFAR_EMOJI,
    emoji_font_size=30,
    heatmap_mode="logit_values",
    alpha_color=0.6,
    return_graph=False,
):
    valid_heatmap_modes = ["logit_values", "emoji_colors"]
    if heatmap_mode not in valid_heatmap_modes:
        raise ValueError(
            f"Invalid heatmap_mode '{heatmap_mode}'. Valid options are {valid_heatmap_modes}."
        )

    if isinstance(image, np.ndarray) and image.shape[-1] == 3:
        # Image is in correct format.
        pass
    elif isinstance(image, torch.Tensor):
        # Convert torch.Tensor to numpy.ndarray if necessary.
        image = image.detach().cpu().numpy().transpose(1, 2, 0)
        if image.max() <= 1.0:
            image *= 255  # Convert from [0, 1] to [0, 255] if needed.
        image = image.astype(np.uint8)

    grid_size_x = image.shape[1] // patch_size
    print(f"grid_size_x: {grid_size_x}")
    grid_size_y = image.shape[0] // patch_size
    print(f"grid_size_y: {grid_size_y}")

    # Initialize matrices with NaNs for logit values or zeros for emoji colors
    if heatmap_mode == "logit_values":
        heatmap_matrix = np.full((grid_size_y, grid_size_x), np.nan)
    else:  # 'emoji_colors'
        heatmap_matrix = np.zeros((grid_size_y, grid_size_x))

    if imagenet_class_to_emoji is not None and heatmap_mode == "emoji_colors":
        # Map emojis to unique integers if in 'emoji_colors' mode
        unique_emojis = {
            emoji: idx
            for idx, emoji in enumerate(set(imagenet_class_to_emoji.values()))
        }
    else:
        unique_emojis = {}

    fig = px.imshow(image)
    fig.update_traces(hoverinfo="none", hovertemplate=None)

    annotations = []
    hover_texts = []
    x_centers = []
    y_centers = []
    for patch_index, patch_data in sorted(patch_dictionary.items()):
        if patch_index == 0:
            continue  # Skip the CLS token.
        logit_data = patch_data[layer_idx]
        length_of_patch_data = len(logit_data)
        if length_of_patch_data >= 3:
            logit_value, class_name, class_index = logit_data[:3]
        else:
            print("Error in length of patch data.")
            continue

        adjusted_index = patch_index - 1
        row, col = divmod(adjusted_index, grid_size_x)
        print(f"row: {row}")
        print(f"col: {col}")

        if heatmap_mode == "logit_values":
            heatmap_matrix[row, col] = logit_value
        else:  # 'emoji_colors'
            emoji = imagenet_class_to_emoji.get(class_index, "❓")
            heatmap_matrix[row, col] = unique_emojis.get(emoji, 0)

        emoji = (
            imagenet_class_to_emoji.get(class_index, "❓")
            if imagenet_class_to_emoji is not None
            else "❓"
        )
        annotations.append(
            go.layout.Annotation(
                x=col * patch_size + patch_size / 2,
                y=row * patch_size + patch_size / 2,
                text=emoji,
                showarrow=False,
                font=dict(size=emoji_font_size),
            )
        )
        hover_texts.append(
            f"Patch: {patch_index}<br>Class: {class_name}<br>Logit: {logit_value:.2f}"
        )
        x_centers.append(col * patch_size + patch_size / 2)
        y_centers.append(row * patch_size + patch_size / 2)

    # Choose color scale based on heatmap_mode
    colorscale = (
        "Viridis" if heatmap_mode == "logit_values" else px.colors.qualitative.Plotly
    )
    heatmap = go.Heatmap(
        z=heatmap_matrix,
        x0=0.5 * patch_size,
        dx=patch_size,
        y0=0.5 * patch_size,
        dy=patch_size,
        colorscale=colorscale,
        opacity=alpha_color,
        showscale=(heatmap_mode == "logit_values"),
        hoverinfo="none",
    )
    fig.add_trace(heatmap)

    fig.add_trace(
        go.Scatter(
            x=x_centers,
            y=y_centers,
            mode="markers",
            marker=dict(color="rgba(0,0,0,0)"),
            hoverinfo="text",
            text=hover_texts,
            hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial"),
        )
    )
    fig.update_layout(annotations=annotations)
    fig.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)

    if return_graph:
        return fig
    else:
        fig.show()


def calculate_accuracy(
    net, data_loader, device, N=2000, batch_size=50, attack_fn=None, **kwargs
):
    net.eval()
    correct = 0
    total = 0
    for items in islice(data_loader, N // batch_size):
        x, labels, *extras = items
        x = x.clone()
        x, labels = x.to(DEVICE), labels.to(DEVICE)

        if attack_fn:
            delta = attack_fn(net, x, labels, **kwargs)
            x = x + delta

        logits = net(x.to(device))
        predictions = torch.argmax(logits, dim=1)
        correct += torch.sum(predictions == labels.to(device)).item()
        total += len(labels)

    return correct / total


def display_patch_logit_lens(
    patch_dictionary,
    width=1000,
    height=1200,
    emoji_size=26,
    return_graph=False,
    show_colorbar=True,
    labels=None,
):
    num_patches = len(patch_dictionary)

    # Assuming data_array_formatted is correctly shaped according to your data structure
    data_array_formatted = np.array(
        [
            [item[0] for item in list(patch_dictionary.values())[i]]
            for i in range(num_patches)
        ]
    )

    # Modify hover text generation based on whether labels are provided
    if labels:
        hover_text = [
            [
                f"{labels[j]}: {item[1]}"
                for j, item in enumerate(list(patch_dictionary.values())[i])
            ]
            for i in range(num_patches)
        ]
    else:
        hover_text = [
            [str(item[1]) for item in list(patch_dictionary.values())[i]]
            for i in range(num_patches)
        ]

    # Creating the interactive heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=data_array_formatted,
            x=list(patch_dictionary.keys())[:num_patches],
            y=[f"{i}" for i in range(data_array_formatted.shape[0])],  # Patch Number
            hoverongaps=False,
            colorbar=dict(title="Logit Value") if show_colorbar else None,
            text=hover_text,
            hoverinfo="text",
        )
    )

    # Initialize a list to hold annotations for emojis
    annotations = []

    # Calculate half the distance between cells in both x and y directions for annotation placement
    x_half_dist = 0.5
    y_half_dist = 0.2

    for i, patch in enumerate(patch_dictionary.values()):
        for j, items in enumerate(
            patch
        ):  # Extract class index directly from the patch_dictionary
            class_index = items[2]
            emoji = CIFAR_EMOJI.get(
                class_index, ""
            )  # Use class index for emoji lookup, default to empty if not found
            if emoji:  # Add annotation if emoji is found
                annotations.append(
                    go.layout.Annotation(
                        x=j + x_half_dist,
                        y=i + y_half_dist,
                        text=emoji,
                        showarrow=False,
                        font=dict(color="white", size=emoji_size),
                    )
                )

    # Add annotations to the figure
    fig.update_layout(annotations=annotations)

    # Configure the layout of the figure
    fig.update_layout(
        title="Per-Patch Logit Lens",
        xaxis=dict(title="Layer Number"),
        yaxis=dict(title="Patch Number"),
        autosize=False,
        width=width,
        height=height,
    )
    fig.show()

    if return_graph:
        return fig
