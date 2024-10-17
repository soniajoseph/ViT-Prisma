import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from resnet import *
from argparse import ArgumentParser
import numpy as np

###############################################################
# The labels of the generated dataset is the same as the
# default torchvision.dataset.CIFAR10()
# To load the dataset:
#   train_set      = torchvision.dataset.CIFAR10(....)
#   train_set.data = torch.load('../Robust/robust_resnet50.pt').mul_(255.0).numpy().tranpose(0, 2, 3, 1).astype(np.uint8)
###############################################################

parser = ArgumentParser()
parser.add_argument('--batch', type=int, default=512)
parser.add_argument('--pgd_steps', type=int, default=1000)
parser.add_argument('--step_size', type=float, default=0.1)
parser.add_argument('--weight', type=str, default="")
parser.add_argument('--save', type=str, default="")
parser.add_argument('--delta', type=float, default=255.0)
parser.add_argument('--normalize', action='store_true')
args = parser.parse_args()

# Prepare dataset
transform_train = transforms.Compose([
    transforms.ToTensor(),
])
trainset = datasets.CIFAR10('./data', train=True, transform=transform_train, download=True)
loader = DataLoader(trainset, batch_size=args.batch, num_workers=8)

# Load Model
model = ResNet50()
model.load_state_dict(torch.load(args.weight))
model = model.cuda()

# Generate Data
target_dataset = []
last_data = None
deltas = []
for data, labels in loader:
    data, labels = data.cuda(), labels.cuda()

    # x_0
    index = np.random.randint(0, len(trainset), [data.shape[0]])
    image_base = []
    for id in index:
        d, l = trainset[id]
        image_base.append(d)
    image_base = torch.stack(image_base, dim=0).cuda()

    target = image_base.clone()
    # Just to clear up gradient
    optimizer = torch.optim.SGD([target], lr=0.0001, momentum=0.9)

    with torch.no_grad():
        train_emb = model.embed_forward(data, args.normalize)

    pbar = tqdm(range(1, args.pgd_steps + 1), ncols=100)
    for j in pbar:
        target.requires_grad = True
        target_emb = model.embed_forward(target, args.normalize)
        loss = torch.norm(train_emb - target_emb)
        optimizer.zero_grad()

        # Updata x^\prime with l-2 PGD
        grad = torch.autograd.grad(loss, target,
                                   retain_graph=False, create_graph=False)[0]
        grad_norms = torch.norm(grad.view(data.shape[0], -1), p=2, dim=1) + 1e-8
        grad = grad / grad_norms.view(data.shape[0], 1, 1, 1)
        target = target.detach() - args.step_size * grad
        delta = target - image_base
        delta_norms = torch.norm(delta.view(data.shape[0], -1), p=2, dim=1)
        factor = args.delta / delta_norms
        factor = torch.min(factor, torch.ones_like(delta_norms))
        delta = delta * factor.view(-1, 1, 1, 1)
        target = torch.clamp(delta + image_base, min=0, max=1).detach()

    if args.save:
        target_dataset += [t.detach().cpu() for t in target]

# Finally save the dataset
target_dataset = torch.stack(target_dataset)

if args.save:
    print('saving files')
    torch.save(target_dataset, args.save)

