import math

import torch


class WarmupThenStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, step_size, gamma=0.5, last_epoch=-1):
        # Does warmup for warmup steps, then moves to step decay parameterized by step_size
        self.warmup_steps = warmup_steps
        self.step_size = step_size
        self.gamma = gamma
        super(WarmupThenStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [base_lr * (self.last_epoch / self.warmup_steps) for base_lr in self.base_lrs]
        else:
            # Check if it's time for step decay
            if (self.last_epoch - self.warmup_steps + 1) % self.step_size == 0:
                current_lr = self.optimizer.param_groups[0]['lr']  # Assuming all parameter groups have the same learning rate
                print(f"Reducing learning rate from {current_lr} to {current_lr * self.gamma} at epoch {self.last_epoch}")

            return [base_lr * (self.gamma ** ((self.last_epoch - self.warmup_steps) // self.step_size))
                    for base_lr in self.base_lrs]


class WarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6, last_epoch=-1):
        """Cosine Annealing with linear warmup learning rate scheduler. It has a linear
        warmup phase for warmup_steps steps and then transitions to cosine annealing.

        Args:
            optimizer: The optimiser for which to schedule the learning rate
            warmup_steps: Number of warmup steps at the start.
            total_steps: Total number of training steps.
            min_lr: Minimum learning rate to anneal to.
            last_epoch: The index of last epoch.
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr

        # Initialize parent class - this creates self.base_lrs from optimizer
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup: go from min_lr to base_lr
            warmup_progress = self.last_epoch / self.warmup_steps
            return [self.min_lr + (base_lr - self.min_lr) * warmup_progress for base_lr in self.base_lrs]
        else:
            # Cosine annealing phase
            progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            return [self.min_lr + (base_lr - self.min_lr) * cosine_factor
                    for base_lr in self.base_lrs]
