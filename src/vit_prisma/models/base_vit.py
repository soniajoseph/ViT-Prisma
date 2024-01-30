import torch
import torch.nn as nn

from vit_prisma.models.layers.patch_embedding import PatchEmbedding
from vit_prisma.models.layers.position_embeddding import PosEmbedding
from vit_prisma.models.layers.layer_norm import LayerNorm, LayerNormPre
from vit_prisma.models.layers.mlp import MLP
from vit_prisma.models.layers.attention import Attention
from vit_prisma.models.layers.transformer_block import TransformerBlock

from vit_prisma.training.training_dictionary import activation_dict, initialization_dict
# from vit_prisma.models.prisma_net import PrismaNet
from src.vit_prisma.prisma.hook_points import HookedRootModule, HookPoint

from src.vit_prisma.configs import HookedViTConfig

from typing import Union, Dict, List, Int, Float, Tuple

class HookedViT(HookedRootModule):
    """
    Base vision model.
    Based on 'An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale' https://arxiv.org/abs/2010.11929.
    Adapted from TransformerLens: https://github.com/neelnanda-io/TransformerLens/blob/main/transformer_lens/HookedTransformer.py
    Inspiration also taken from the timm library.
    """

    def __init__(
            self,
            cfg: HookedViTConfig,
            move_to_device: bool = True,
    ):
        """
        Model initialization
        """

        super().__init__()
        if isinstance(cfg, Dict):
            cfg = HookedViTConfig(**cfg)
        elif isinstance(cfg, str):
            raise ValueError(
                "Please pass in a config dictionary or HookedViT object. If you want to load a "
                "pretrained model, use HookedViT.from_pretrained() instead."
            )
        self.cfg = cfg

        # Patch embeddings
        self.embed = PatchEmbedding(self.cfg)
        self.hook_embed = HookPoint()

        # Position embeddings
        self.pos_embed = PosEmbedding(self.cfg)
        self.hook_pos_embed = HookPoint()

        # Blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(self.cfg, block_index)
                for block_index in range(self.cfg.transformer.num_layers)
            ]
        )

        # Final layer norm
        if self.cfg.normalization_type == "LN":
            self.ln_final = LayerNorm(self.cfg)
        elif self.cfg.normalization_type == "LNPre":
            self.ln_final = LayerNormPre(self.cfg)
        elif self.cfg.normalization_type is None:
            self.ln_final = nn.Identity()
        else:
            raise ValueError(f"Invalid normalization type: {self.cfg.normalization_type}")

        # Final classification head
        self.head = nn.Linear(self.cfg.d_model, self.cfg.n_classes)

        # Initialize weights
        self.init_weights()

        # Set up HookPoints
        self.setup()

    def forward(
            self,
            input: Union[
                str,
                List[str],
                Int[torch.Tensor, "batch pos"],
                Float[torch.Tensor, "batch pos d_model"],
            ],
    ) -> Union[
        None,
        Float[torch.Tensor, "batch n_classes"],
    ]:
        
        # Embedding
        embed = self.hook_embed(self.embed(input))

        # Position embedding
        pos_embed = self.hook_pos_embed(self.pos_embed(x))

        residual = embed + pos_embed

        # Blocks
        for block in self.blocks:
            residual = block(residual)

        # Final layer norm
        x = self.ln_final(residual)

        if self.config.classification_type == 'gaap':  # GAAP
            x = x.mean(dim=1)
        elif self.config.classification_type == 'cls':  # CLS token
            x = x[:, 0]
            
        return x if self.cfg.return_type == 'pre_logits' else self.head(x)

    def init_weights(self):
        if self.cfg.classification_type == 'cls':
            nn.init.normal_(self.cls_token, std=self.config.init.cls_std)
        nn.init.trunc_normal_(self.position_embedding, std=self.config.init.pos_std)   
        if self.config.init.weight_type == 'he':
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, nonlinearity=initialization_dict[self.cfg.activation_name])
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

