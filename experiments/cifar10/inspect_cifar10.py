from experiments.utils.visualise import plot_image
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from vit_prisma.models.base_vit import HookedViT
from vit_prisma.utils.load_model import load_model as load_remote_model
from experiments.attack.attacks import PGD
from vit_prisma.utils.constants import MODEL_CHECKPOINTS_DIR, DEVICE, DATA_DIR
from vit_prisma.utils.data_utils.cifar.cifar_10_utils import load_cifar_10

from experiments.cifar10.cifar10_config import CIFAR10_CONFIG
from experiments.utils.load_model import load_prisma_model
from experiments.utils.performance_utils import calculate_accuracy

######## Load local model #########

checkpoint_path = MODEL_CHECKPOINTS_DIR / "cifar10-clean" / "model_1987392.pth"
model = load_prisma_model(CIFAR10_CONFIG, checkpoint_path)

######## Load huggingface model #########

# model = HookedViT.from_pretrained("open-clip:laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K", is_timm=False, is_clip=True).to(DEVICE)
# print(model.cfg)

######## Load data #############

train_dataset, val_dataset, test_dataset = load_cifar_10(
    DATA_DIR / "cifar10",
    image_size=model.cfg.image_size,
    augmentation=False,
    with_index=False,
    visualisation=False,
)
train_dataloader = DataLoader(train_dataset, batch_size=64, sampler=None, shuffle=False)
val_dataloader = DataLoader(val_dataset, batch_size=64, sampler=None, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=64, sampler=None, shuffle=False)

model.eval()
# print(model)

#######  Calculate the clean accuracy of the model  ##############

train_accuracy = calculate_accuracy(model, train_dataloader, device=DEVICE)
print(f"The train accuracy is: {train_accuracy}")

val_accuracy = calculate_accuracy(model, val_dataloader, device=DEVICE)
print(f"The val accuracy is: {val_accuracy}")

test_accuracy = calculate_accuracy(model, test_dataloader, device=DEVICE)
print(f"The test accuracy is: {test_accuracy}")

# TODO EdS: I accidentally trained on visualisation images.

###### Calculate the robust accuracy of the model #######

# attack_type = "PGD"
# norm = "Linf"
# epsilon = 0.1
# step = 0
#
# if attack_type == 'AA':
#     test_set = val_dataset
#     data = [test_set[i][0] for i in range(0, 5000)]
#     data = torch.stack(data).cuda()
#     target = [test_set[i][1] for i in range(0, 5000)]
#     target = torch.LongTensor(target).cuda()
#     adversary = AutoAttack(model, norm=args.norm, eps=args.epsilon, log_path=args.log_path, version='standard')
#     adv_complete = adversary.run_standard_evaluation(data, target, bs=args.batch_size)
#
# for epsilon in [0.3, 0.5, 1.0, 2.0, 3.0]:
#     if attack_type == 'PGD':
#         if norm == 'L2':
#             # adversary = L2PGDAttack(model, epsilon=epsilon, alpha=0.1, iteration=step)
#             # adversary = PGD(model, eps=epsilon, alpha=0.1, steps=step, range=(-2, 2))
#             pass
#         else:
#             adversary = PGD(model, eps=epsilon, alpha=0.1, steps=step, range=(-3, 3))
#             # adversary = LinfPGDAttack(model, epsilon=epsilon, alpha=1 / 255, iteration=step)
#             # adversary = LinfPGDAttack(model, epsilon=epsilon, alpha=1 / 255, iteration=step)
#     else:
#         adversary = attack.CW(model, steps=step)
#
#     correct = 0
#     count = 0
#     idx = 0
#     for image, label in tqdm(test_dataloader):
#         plot_image(image[0,...], save_path=DATA_DIR / f"{idx}.jpg")
#         adv_img = adversary(image.to(DEVICE), label.to(DEVICE))
#         plot_image(adv_img[0, ...], save_path=DATA_DIR / f"{idx}_adv.jpg")
#         logits = model(adv_img, save_path=)
#         correct += (logits.argmax(axis=1).cpu() == label).sum().cpu().item()
#         count += logits.shape[0]
#
#     print(f'For epsilon: {epsilon} the robust accuracy is {correct / len(test_dataset)} ({correct} out of {len(test_dataset)})')

