import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import copy


class L2PGDAttack(object):
    def __init__(self, model, epsilon, alpha, iteration):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.k = iteration

    def perturb(self, x_natural, y):
        batch_size = x_natural.shape[0]
        x = x_natural.detach()
        x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        for i in range(self.k):
            x.requires_grad_()
            with torch.enable_grad():
                logits = self.model(x)
                loss = F.cross_entropy(logits, y)
            grad = torch.autograd.grad(loss, [x])[0]
            grad_norms = torch.norm(grad.view(batch_size, -1), p=2, dim=1) + 1e-8
            grad = grad / grad_norms.view(batch_size, 1, 1, 1)
            x = x.detach() - self.alpha * grad
            delta = x - x_natural
            delta_norms = torch.norm(delta.view(batch_size, -1), p=2, dim=1)
            factor = self.epsilon / delta_norms
            factor = torch.min(factor, torch.ones_like(delta_norms))
            delta  = delta * factor.view(-1, 1, 1, 1)
            x = torch.clamp(delta + x_natural, min=0, max=1).detach()
        return x
    
class LinfPGDAttack(object):
    def __init__(self, model, epsilon, alpha, iteration):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.k = iteration

    def perturb(self, x_natural, y):
        x = x_natural.detach()
        x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        for i in range(self.k):
            x.requires_grad_()
            with torch.enable_grad():
                logits = self.model(x)
                loss = F.cross_entropy(logits, y)
            grad = torch.autograd.grad(loss, [x])[0]
            x = x.detach() - self.alpha * grad
            delta = x - x_natural
            delta = torch.clamp(delta, -self.epsilon, self.epsilon)
            x = torch.clamp(delta + x_natural, min=0, max=1).detach()
        return x


attack_methods = {
    "l2": L2PGDAttack,
    "linf": LinfPGDAttack,
}
