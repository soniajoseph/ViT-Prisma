import torchattacks as attack
from experiments.attack.auto_attack.autoattack import AutoAttack
import torch
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm

from vit_prisma.utils.constants import DATA_DIR, MODEL_CHECKPOINTS_DIR
from vit_prisma.utils.data_utils.cifar.cifar_10_utils import load_cifar_10

from experiments.cifar10.cifar10_config import CIFAR10_CONFIG
from experiments.utils.load_model import load_prisma_model

checkpoint_path = MODEL_CHECKPOINTS_DIR / "cifar100" / "clean" / "model_2038400.pth"
cfg = CIFAR10_CONFIG
norm = "L2"
attack = "PGD"
epsilon = 0.1
step = 10

model = load_prisma_model(CIFAR10_CONFIG, checkpoint_path)
train_data, val_data, test_data = load_cifar_10(DATA_DIR / "cifar", augmentation=True)
dataloader = DataLoader(test_data, batch_size=32, shuffle=False)

if attack == 'AA':
    test_set = val_dataset
    data = [test_set[i][0] for i in range(0, 5000)]
    data = torch.stack(data).cuda()
    target = [test_set[i][1] for i in range(0, 5000)]
    target = torch.LongTensor(target).cuda()
    adversary = AutoAttack(model, norm=args.norm, eps=args.epsilon, log_path=args.log_path,
                           version='standard')
    adv_complete = adversary.run_standard_evaluation(data, target, bs=args.batch_size)

if attack == 'PGD':
    if norm == 'L2':
        adversary = attack.PGDL2(model, eps=epsilon, alpha=0.1, steps=step)
    else:
        adversary = attack.PGD(model, eps=args.epsilon, alpha=1 / 255, steps=args.step)
else:
    adversary = attack.CW(model, steps=args.step)

cnt, correct = 0, 0
for image, label in tqdm(dataloader):
    adv_img = adversary(image.cuda(), label.cuda())
    logits = model(adv_img)
    correct += (logits.argmax(axis=1).cpu() == label).sum().cpu().item()

    print(correct, image.shape[0])

    cnt += 1
    if cnt == 20:
        break

print(f'robust acc = {correct / 5000}')

# adversary = attack.PGD(model, eps=0.0157, alpha=1 / 255, steps=args.step)
# cnt, correct = 0, 0
# for image, label in tqdm(val_dataloader):
#     adv_img = adversary(image.cuda(), label.cuda())
#     logits = model(adv_img)
#     correct += (logits.argmax(axis=1).cpu() == label).sum().cpu().item()
#
#     print(correct, image.shape[0])
#
#     cnt += 1
#     if cnt == 20:
#         break
#
# print(f'robust acc = {correct / 5000}')