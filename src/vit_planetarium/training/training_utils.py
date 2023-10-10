import torch
import torch.nn as nn
from itertools import islice
import numpy as np
import random
from vit_planetarium.training.training_dictionary import loss_function_dict


def calculate_accuracy(net, data_loader, device, N=2000, batch_size=50):
    net.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for items in islice(data_loader, N // batch_size):
            x, labels, *extras = items
            logits = net(x.to(device))
            predictions = torch.argmax(logits, dim=1)
            correct += torch.sum(predictions == labels.to(device)).item()
            total += len(labels)
        return correct / total

def calculate_loss(net, data_loader, loss_fn, device, N=2000, batch_size=50):
    net.eval()
    with torch.no_grad():
        total = 0
        points = 0
        for items in islice(data_loader, N // batch_size):
            x, labels, *extras = items
            logits = net(x.to(device))
            total += loss_fn(logits, labels.to(device)).item()
            points += len(labels)
        return total / points
    
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)