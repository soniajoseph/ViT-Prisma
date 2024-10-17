import torch
import torchattacks
from torch.utils.data import DataLoader
from tqdm import tqdm
from vit_prisma.utils.constants import MODEL_CHECKPOINTS_DIR, DEVICE, DATA_DIR
from vit_prisma.utils.data_utils.cifar.cifar_10_utils import load_mnist

from experiments.attack.PGD import LinfPGDAttack
from experiments.attack.attacks import PGD
from experiments.attack.auto_attack.autoattack import AutoAttack
from experiments.mnist.mnist_config import MNIST_CONFIG
from experiments.utils.load_model import load_prisma_model
from experiments.utils.performance_utils import calculate_accuracy


checkpoint_path = MODEL_CHECKPOINTS_DIR / "mnist-clean" / "model_1378816.pth"
cfg = MNIST_CONFIG

model = load_prisma_model(cfg, checkpoint_path)

train_data, val_data, test_data = load_mnist(DATA_DIR / "mnist", augmentation=False, visualisation=False)

model.eval()
print(model)

###### Calculate the clean accuracy of the model ######

test_loader = DataLoader(test_data, batch_size=cfg.batch_size, shuffle=False)
accuracy = calculate_accuracy(model, test_loader, device=DEVICE)
print(f"The test accuracy is: {accuracy}")

###### Calculate the robust accuracy of the model #######

attack_type = "PGD"
norm = "L2"
epsilon = 2.0
step = 0

if attack_type == 'AA':
    test_set = val_dataset
    data = [test_set[i][0] for i in range(0, 5000)]
    data = torch.stack(data).cuda()
    target = [test_set[i][1] for i in range(0, 5000)]
    target = torch.LongTensor(target).cuda()
    adversary = AutoAttack(model, norm=args.norm, eps=args.epsilon, log_path=args.log_path, version='standard')
    adv_complete = adversary.run_standard_evaluation(data, target, bs=args.batch_size)

if attack_type == 'PGD':
    if norm == 'L2':
        # adversary = L2PGDAttack(model, epsilon=epsilon, alpha=0.1, iteration=step)
        adversary = PGD(model, eps=epsilon, alpha=0.1, steps=step, range=(-2, 2))
    else:
        adversary = LinfPGDAttack(model, epsilon=epsilon, alpha=1 / 255, iteration=step)
        adversary = LinfPGDAttack(model, epsilon=epsilon, alpha=1 / 255, iteration=step)
else:
    adversary = attack.CW(model, steps=step)

correct = 0
count = 0
for image, label in tqdm(test_loader):
    adv_img = adversary(image.to(cfg.device), label.to(cfg.device))
    logits = model(adv_img)
    correct += (logits.argmax(axis=1).cpu() == label).sum().cpu().item()
    count += logits.shape[0]

print(f'The robust accuracy is {correct / len(test_data)} ({correct} out of {len(test_data)})')
