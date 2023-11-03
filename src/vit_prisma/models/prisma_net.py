import torch
import torch.nn as nn

class PrismaNet(nn.Module):
    def __init__(self):
        super().__init__()

    def get_activations(self, images: torch.Tensor):
        
        activations = {}

        def save_activation(name):
            def hook(model, input, output):
                activations[name] = output.detach()
            return hook
        
        active_hooks = []

        for name, layer in self.named_modules():
            active_hook = layer.register_forward_hook(save_activation(name))
            active_hooks.append(active_hook)
        
        self.forward(images)

        for hook in active_hooks:
            hook.remove()

        return activations