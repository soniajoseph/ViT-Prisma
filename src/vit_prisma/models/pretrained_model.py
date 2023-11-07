import torch.nn as nn
from transformers import ViTForImageClassification, AutoImageProcessor
import timm
from vit_prisma.models.prisma_net import PrismaNet

class PretrainedModel(PrismaNet):
    '''
    Wrapper class for pretrained models from huggingface and timm, 
    which can be used for finetuning with the trainer available in the Prisma project.
    '''
    def __init__(self, pretrained_model_name: str, config, is_timm: bool = False):
        super().__init__()
        self.is_timm = is_timm
        
        self.pretrained_model_name = pretrained_model_name
        self.n_classes = None if config is None else config.classification.num_classes
        if is_timm:
            self.model = timm.create_model(pretrained_model_name, pretrained=True, num_classes=self.n_classes)
        else:
            self.image_processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
            self.model = ViTForImageClassification.from_pretrained(pretrained_model_name)

            self.vit = self.model.vit

            if config is None or self.n_classes == self.model.classifier.out_features:
                self.classifier = self.model.classifier
            else:
                print(f"Initializing new classifier head for finetuning with shape: ({self.model.classifier.in_features, self.n_classes})")
                self.classifier = nn.Linear(self.model.classifier.in_features, self.n_classes)
    
    def __getattr__(self, name):
        try:
            return self._modules.get('model').__getattr__(name)
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' does not have '{name}'")

    def forward(self, x):
        if self.is_timm:
            x = self.model(x)
        else:
            x = self.image_processor(x, return_tensors="pt").pixel_values
            x = self.vit(x).last_hidden_state[:, 0, :]
            x = self.classifier(x)
        
        return x
    
    def convert_hf_model_config(self):
        if self.is_timm:
            raise NotImplementedError
        else:
            hf_config = self.vit.config
            prisma_config = {
                'image_config': {
                    'image_size': hf_config.image_size,
                    'patch_size': hf_config.patch_size,
                    'n_channels': hf_config.num_channels,
                },
                'transformer_config': {
                    'hidden_dim': hf_config.hidden_size,
                    'num_heads': hf_config.num_attention_heads,
                    'num_layers': hf_config.num_hidden_layers,
                    'mlp_dim': hf_config.intermediate_size,
                    'activation_name': hf_config.hidden_act,
                },
                'layer_norm_config': {
                    'qknorm': None,
                    'layer_norm_eps': hf_config.layer_norm_eps,
                },
                'dropout_config': {
                    'patch': hf_config.hidden_dropout_prob,
                    'position': hf_config.hidden_dropout_prob,
                    'attention': hf_config.attention_probs_dropout_prob,
                    'proj': None,
                    'mlp': None,
                },
                'classification_config': {
                    'num_classes': self.classifier.out_features,
                },
            }

            return prisma_config
