import torch
import torch.nn as nn
from src.layers.transformer_block import TransformerBlock

class BaseViT(nn.Module):
    """
    Base vision model.
    Based on'`An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale' https://arxiv.org/abs/2010.11929.
    Inspiration also taken from the timm library.
    """
    def __init__(self, config):
        super(BaseViT, self).__init__()
        
        self.config = config
        self.patch_embedding = embed_fn(self.config)
        self.patch_dropout = nn.Dropout(self.config.patch_dropout)

        self.cls_token  = nn.Parameter(torch.zeros(1, 1, self.config.hidden_size)) if not self.config.global_pool
        token_length = num_patches + 1 if not self.config.global_pool else num_patches
        self.position_embedding = nn.Parameter(torch.randn(1, token_length, self.config.hidden_size))
        self.position_dropout = nn.Dropout(self.config.position_dropout)
        block_fn = self.config.block_fn if hasattr(self.config, 'block_fn') else TransformerBlock
        self.blocks = nn.Sequential(*[block_fn(self.config) for l in range(self.config.num_layers)])
        self.head = nn.Linear(self.config.hidden_size, self.config.num_classes)

        self.init_weights(self.config.weight_init_type)

    def init_weights(self, weight_init_type: str):
        if not self.config.global_pool:
            nn.init.normal_(self.cls_token, std=self.config.cls_std_init)
        nn.init.trunc_normal_(self.position_embedding, std=self.config.pos_std_init)   
        if weight_init_type == 'he':
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, nonlinearity=self.config.activation_name) # activation_fn_name = 'relu'
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x, pre_logits: bool = False):
        x = self.patch_embedding(x)
        x = self.patch_dropout(x) if self.config.patch_dropout > 0 else x
        x = self.position_embedding(x)
        x = self.position_dropout(x) if self.config.position_dropout > 0 else x
        x = self.blocks(x)
        if self.config.global_pool: # GAAP
            x = x.mean(dim=1)
        else: # CLS token
            x = x[:, 0]
        x = self.norm(x)
        return x if pre_logits else self.head(x)
    