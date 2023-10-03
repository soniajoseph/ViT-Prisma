import torch
import torch.nn as nn
from itertools import islice
import numpy as np
import random


def calculate_accuracy(net, data_loader, device, N=2000, batch_size=50):
    net.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for x, labels, _ in islice(data_loader, N // batch_size):
            logits = net(x.to(device))
            predictions = torch.argmax(logits, dim=1)
            correct += torch.sum(predictions == labels.to(device)).item()
            total += len(labels)
        return correct / total

def calculate_loss(net, data_loader, loss_function, device, N=2000, batch_size=50):
    net.eval()
    with torch.no_grad():
        loss_fn = loss_function_dict[loss_function](reduction='sum')
        one_hots = torch.eye(10, 10).to(device) # what is this?? 
        total = 0
        points = 0
        for x, labels, _ in islice(data_loader, N // batch_size):
            logits = net(x.to(device))
            if loss_function == 'CrossEntropy':
                total += loss_fn(logits, labels.to(device)).item()
            elif loss_function == 'MSE':
                total += loss_fn(logits, one_hots[labels.to(device)]).item()
            points += len(labels)
        return total / points
    
def set_seed(seed, dtype):
    torch.set_default_dtype(dtype)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)