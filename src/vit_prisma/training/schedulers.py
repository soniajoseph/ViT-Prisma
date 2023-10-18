
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