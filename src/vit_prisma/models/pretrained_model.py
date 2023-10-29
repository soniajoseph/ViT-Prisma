import torch.nn as nn
from transformers import ViTModel, AutoImageProcessor


class PretrainedModel(nn.Module):
    def __init__(self, pretrained_model: str, config):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.image_processor = AutoImageProcessor.from_pretrained(pretrained_model)
        self.n_classes = config.classification.num_classes
        self.net = ViTModel.from_pretrained(pretrained_model)
        self.classifier = nn.Linear(self.net.config.hidden_size, self.n_classes)
        

    def forward(self, x):
        x = self.image_processor(x, return_tensors="pt").pixel_values
        x = self.net(x).last_hidden_state[:, 0, :]
        x = self.classifier(x)
        return x
