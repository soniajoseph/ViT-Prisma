from itertools import islice
import os

import torch
from sympy import pretty_print
from torch.utils.data import DataLoader

from experiments.model_training.imagenet100_config import IMAGENET100_VIT_CONFIG_FAST
from vit_prisma.models.base_vit import HookedViT

# from src.vision.utils.attack import run_adversarial_attack, pgd_l2_adv
# from src.vision.utils.constants import (
#     DEVICE,
#     VIT_CONFIG,
#     CHECKPOINTS_DIR,
#     CIFAR_DIR,
# )
# from src.vision.utils.utils import get_data, calculate_accuracy
# from src.vision.utils.visualise import plot_image

from vit_prisma.utils.data_utils.cifar.cifar_10_utils import load_imagenet100

# checkpoint_path = CHECKPOINTS_DIR / "cifar10-clean" / "model_4090960.pth"

# model_function = HookedViT
# model = model_function(VIT_CONFIG)
#
#
# if os.path.exists(checkpoint_path):
#     checkpoint = torch.load(checkpoint_path, map_location=torch.device(DEVICE))
#     print([k for k in checkpoint["model_state_dict"].keys()])
#     model.load_state_dict(checkpoint["model_state_dict"])
#     model = model.to(DEVICE)

train_dataset, val_dataset = load_imagenet100(IMAGENET100_VIT_CONFIG_FAST.image_size)
train_dataloader = DataLoader(train_dataset, batch_size=128, sampler=None, shuffle=True)
test_dataloader = DataLoader(val_dataset, batch_size=128, sampler=None, shuffle=False)

# model.eval()
# print(model)

# Calculate the clean accuracy of the model
# accuracy = calculate_accuracy(model, test_dataloader, DEVICE, len(test), batch_size=128)
# print(f"The accuracy is: {accuracy}")

# TODO EdS: Implement smooth attack so there are not obvious patches in the adversary

for items in islice(test_dataloader, len(test) // 128):
    x, labels, *extras = items
    x = x.clone()
    x, labels = x.to(DEVICE), labels.to(DEVICE)
    delta = pgd_l2_adv(model, x, labels, **attack_params)
    x = x + delta
    break

for image_idx in range(10):
    image, label = test[image_idx]
    attacked_image = x[image_idx, ...].unsqueeze(0)
    image, attacked_image = image.detach().cpu(), attacked_image.detach().cpu()

    plot_image(image, CIFAR_DIR / f"img/img_{image_idx}.png")
    plot_image(
        attacked_image.squeeze(0), CIFAR_DIR / f"img/img_{image_idx}_attacked.png"
    )

    image, attacked_image = image.to(DEVICE), attacked_image.to(DEVICE)

    print(f"The label is: {label}")
    output, cache = model.run_with_cache(image.unsqueeze(0))
    attacked_output, attacked_cache = model.run_with_cache(attacked_image)
    print(f"The output is: {torch.argmax(output)}")
