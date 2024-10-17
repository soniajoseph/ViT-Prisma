"""Helpers relevant to training models."""
from itertools import islice

import torch
from vit_prisma.training.training_utils import PrismaCallback
from vit_prisma.utils.constants import DEVICE


class DemoCallback(PrismaCallback):
    def on_epoch_end(self, epoch, net, val_loader, wandb_logger):
        print(f"You reached the end of epoch: {epoch}")

    def on_step_end(self, step, net, val_loader, wandb_logger):
        if step % 200 == 0:
            net.eval()
            with torch.no_grad():
                for items in islice(val_loader, 2000 // 256):
                    x, labels, *extras = items
                    logits = net(x.to(DEVICE))
                    loss = torch.nn.CrossEntropyLoss()(
                        logits, labels.to(DEVICE)
                    ).item()
                    print(f"Loss: {loss}")
                    print(f"Labels: {labels}")
                    print(f"Preds: {torch.argmax(logits, dim=-1)}")
                    print(f"logits: {logits}")
                    break
