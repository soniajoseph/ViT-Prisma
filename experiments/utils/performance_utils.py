from itertools import islice

import torch
from tqdm import tqdm

from vit_prisma.utils.constants import DEVICE


def calculate_accuracy(
    net, data_loader, device, N=2000, batch_size=50, attack_fn=None, **kwargs
):
    net.eval()
    correct = 0
    total = 0

    if N:
        d = islice(data_loader, N // batch_size)
        t = N // batch_size
    else:
        d = data_loader
        t = len(data_loader)

    for items in tqdm(d, total=t):
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

    print(f"Number of correct preds: {correct} our of {total}")
    return correct / total