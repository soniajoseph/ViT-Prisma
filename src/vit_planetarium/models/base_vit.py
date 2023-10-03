import torch
import torch.nn as nn
from vit_planetarium.models.layers.transformer_block import TransformerBlock
from vit_planetarium.models.layers.patch_embedding import PatchEmbedding

class BaseViT(nn.Module):
    """
    Base vision model.
    Based on'`An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale' https://arxiv.org/abs/2010.11929.
    Inspiration also taken from the timm library.
    """
    def __init__(self, config, logger = None):
        super(BaseViT, self).__init__()


        self.logger = logger
        self.config = config


        layer_norm = self.config.LayerNorm.layer_norm_eps
        hidden_dim = self.config.Transformer.hidden_dim

        
        self.patch_embedding = embed_fn(self.config) if hasattr(self.config, 'embed_fn') else PatchEmbedding(self.config)
        self.patch_dropout = nn.Dropout(self.config.Dropout.patch)

        self.cls_token  = nn.Parameter(torch.zeros(1, 1, hidden_dim)) if not self.config.global_pool else None
        num_patches = (self.config.image_size // self.config.patch_size)**2
        token_length = num_patches + 1 if not self.config.global_pool else num_patches
        self.position_embedding = nn.Parameter(torch.randn(1, token_length, hidden_dim))
        self.position_dropout = nn.Dropout(self.config.Dropout.position)
        block_fn = self.config.Transformer.block_fn if hasattr(self.config.Transformer, 'block_fn') else TransformerBlock
        self.blocks = nn.Sequential(*[block_fn(self.config) for l in range(self.config.Transformer.num_layers)])
        self.head = nn.Linear(hidden_dim, self.config.num_classes)

        self.pre_head_norm = nn.LayerNorm(hidden_dim, eps=layer_norm) if layer_norm > 0 else nn.Identity()
        self.pre_block_norm = nn.LayerNorm(hidden_dim, eps=layer_norm) if layer_norm > 0 else nn.Identity()

        self.init_weights()

    def init_weights(self):
        if not self.config.global_pool:
            nn.init.normal_(self.cls_token, std=self.config.Initialization.cls_std)
        nn.init.trunc_normal_(self.position_embedding, std=self.config.Initialization.pos_std)   
        if self.config.Initialization.weight_type == 'he':
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, nonlinearity=self.config.Transformer.activation_name) #
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x, pre_logits: bool = False):
        x = self.patch_embedding(x)
        x = self.patch_dropout(x) if self.config.Dropout.patch > 0 else x
        if not self.config.global_pool:
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.position_embedding
        x = self.position_dropout(x) if self.config.Dropout.position > 0 else x
        x = self.pre_block_norm(x)
        x = self.blocks(x)
        if self.config.global_pool: # GAAP
            x = x.mean(dim=1)
        else: # CLS token
            x = x[:, 0]
        x = self.pre_head_norm(x)
        return x if pre_logits else self.head(x)
    