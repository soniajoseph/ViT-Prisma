# """
# Code to save logits of all imagenet datapoints for future analysis.

# """

# import timm
# import torch
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader
# import h5py
# import numpy as np
# import logging
# import torch.nn.functional as F

# from tqdm.auto import tqdm

# import os

# from vit_prisma.circuits.timmAblation import timmAblation

# # Setup basic logging
# logging.basicConfig(level=logging.INFO)

# # 1. Load and preprocess the ImageNet dataset
# data_transforms = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])


# # CHANGE these to your own
# imagenet_path = '/network/datasets/imagenet.var/imagenet_torchvision/val/'
# save_dir = '/network/scratch/s/sonia.joseph/imagenet_logits'
# vanilla = False


# imagenet_data = datasets.ImageFolder(imagenet_path, transform=data_transforms)
# data_loader = DataLoader(imagenet_data, batch_size=64, shuffle=False)
# logging.info(f"ImageNet dataset loaded and preprocessed. Total datapoints: {len(imagenet_data)}")

# # cycle through all attn and mlp ablations and rerun

# # 2. Load a pre-trained model from timm
# model_name = 'vit_base_patch32_224'
# if vanilla:
#     model = timm.create_model(model_name, pretrained=True)
# else:
#     model = timmAblation(model_name)

# num_layers = 12
# for layer_idx in range(num_layers):

#     logging.info(f"On layer {layer_idx}")

#     model.eval()
#     model = model.cuda()

#     # Ablate model
#     model.ablate_mlp_of_block(layer_idx)

#     correct_predictions = 0
#     total_predictions = 0

#     # Determine the output size of your model
#     output_size = model(torch.randn(1, 3, 224, 224).cuda()).shape[1]

#     save_path = os.path.join(save_dir, f'layer{layer_idx}_mlp')
#     os.makedirs(save_path, exist_ok=True)


#     # Open an HDF5 file to save logits and labels
#     with h5py.File(os.path.join(save_path, f'logits_and_labels.h5'), 'w') as h5f:
#         # Create a dataset to store logits
#         logits_dataset = h5f.create_dataset('logits', shape=(1, output_size), maxshape=(None, output_size), chunks=True)
#         # Create a dataset to store labels
#         labels_dataset = h5f.create_dataset('labels', shape=(1,), maxshape=(None,), dtype='i', chunks=True)
#         logging.info("HDF5 file created for storing logits and labels.")

#         processed_datapoints = 0
#         for batch_num, (inputs, labels) in tqdm(enumerate(data_loader), total=len(data_loader)):

#             with torch.no_grad():
#                 inputs, labels = inputs.cuda(), labels.cuda()
#                 outputs = model(inputs)

#                 probabilities = F.softmax(outputs, dim=1)
#                 predicted_labels = probabilities.argmax(dim=1)

#                 correct_predictions += (predicted_labels == labels).sum().item()
#                 total_predictions += labels.size(0)

#                 logits = outputs.cpu().numpy()
#                 labels = labels.cpu().numpy()

#                 batch_size = logits.shape[0]
#                 logits_dataset.resize((processed_datapoints + batch_size), axis=0)
#                 logits_dataset[processed_datapoints:] = logits
#                 labels_dataset.resize((processed_datapoints + batch_size), axis=0)
#                 labels_dataset[processed_datapoints:] = labels
#                 processed_datapoints += batch_size

#     #         logging.info(f"Processed batch {batch_num + 1}/{len(data_loader)}, Total datapoints processed: {processed_datapoints}")

#     total_accuracy = correct_predictions / total_predictions
#     logging.info(f"All logits and labels saved successfully. Total accuracy: {total_accuracy:.4f}")
#     with open(os.path.join(save_path, 'accuracy.txt'), 'w') as acc_file:
#         acc_file.write(f"Accuracy for Layer {layer_idx},{total_accuracy:.4f}")

#     del model
#     model = timmAblation(model_name)

