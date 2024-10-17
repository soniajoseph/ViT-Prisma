import torch
import torch.nn as nn
from itertools import islice
import numpy as np
import random

from transformers import ViTConfig
from vit_prisma.configs.HookedViTConfig import HookedViTConfig
from vit_prisma.training.training_dictionary import loss_function_dict
from abc import ABC, abstractmethod

class PrismaCallback(ABC):
    @abstractmethod
    def on_epoch_end(self, epoch, net, val_loader, wandb_logger):
        pass

    @abstractmethod
    def on_step_end(self, step, net, val_loader, wandb_logger):
        pass

def calculate_accuracy(net, data_loader, cfg: HookedViTConfig, N=2000, batch_size=50):
    net.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for items in islice(data_loader, N // batch_size):
            x, labels, *extras = items
            if cfg.attack_method:
                adversary = attack_methods[config.attack_method](model, cfg.attack_epsilon, cfg.attack_alpha, cfg.attack_num_iters)
                x = adversary.perturb(x, labels)
            logits = net(x.to(cfg.device))
            predictions = torch.argmax(logits, dim=1)
            if len(labels.shape) > 1:
                labels = torch.argmax(labels, dim=1)
            correct += torch.sum(predictions == labels.to(cfg.device)).item()
            total += len(labels)
        return correct / total

def calculate_loss(net, data_loader, loss_fn, cfg: HookedViTConfig, N=2000, batch_size=50):
    net.eval()
    with torch.no_grad():
        total = 0
        points = 0
        for items in islice(data_loader, N // batch_size):
            x, labels, *extras = items
            if cfg.attack_method:
                adversary = attack_methods[config.attack_method](model, cfg.attack_epsilon, cfg.attack_alpha, cfg.attack_num_iters)
                x = adversary.perturb(x, labels)
            logits = net(x.to(cfg.device))
            total += loss_fn(logits, labels.to(cfg.device)).item()
            points += len(labels)
        return total / points
    
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)