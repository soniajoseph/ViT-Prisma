import os

import torch
from torch.utils.data import DataLoader

from experiments.model_training.cifar100.cifar100_config import CIFAR100_CONIG
from experiments.utils.performance_utils import calculate_accuracy
from vit_prisma.models.base_vit import HookedViT
from vit_prisma.utils.constants import MODEL_CHECKPOINTS_DIR, DEVICE, DATA_DIR
from vit_prisma.utils.data_utils.cifar.cifar_10_utils import load_cifar_100

checkpoint_path = MODEL_CHECKPOINTS_DIR / "cifar100" / "clean" / "model_2038400.pth"

model_function = HookedViT
model = model_function(CIFAR100_CONIG)


if os.path.exists(checkpoint_path):
    print(f"Loading path at: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(DEVICE))
    print([k for k in checkpoint["model_state_dict"].keys()])
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)

train_dataset, val_dataset, test_dataset = load_cifar_100(
    DATA_DIR / "cifar100",
    image_size=CIFAR100_CONIG.image_size,
    augmentation=False,
)
test_dataloader = DataLoader(test_dataset, batch_size=128, sampler=None, shuffle=False)
val_dataloader = DataLoader(val_dataset, batch_size=128, sampler=None, shuffle=False)

model.eval()
print(model)

# Calculate the clean accuracy of the model
accuracy = calculate_accuracy(model, test_dataloader, device=DEVICE, N=len(test_dataset), batch_size=128)
print(f"The test accuracy is: {accuracy}")
accuracy = calculate_accuracy(model, val_dataloader, device=DEVICE, N=len(val_dataset), batch_size=128)
print(f"The val accuracy is: {accuracy}")
