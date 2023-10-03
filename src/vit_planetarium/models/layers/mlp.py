import torch.nn as nn

class MLP(nn.Module):

    def __init__(self, config, logger = None):
        super().__init__()
        self.logger = logger
        self.config = config

        hidden_dim = self.config.Transformer.hidden_dim
        mlp_dim = self.config.Transformer.mlp_dim

        self.fc1 = nn.Linear(hidden_dim, mlp_dim)
        self.dropout1 = nn.Dropout(self.config.Dropout.mlp) if self.config.Dropout.mlp > 0 else nn.Identity()
        self.act_fn = self.config.Transformer.activation_fn()
        self.mlp_norm = nn.LayerNorm(mlp_dim, eps=self.config.LayerNorm.layer_norm_eps) if self.config.LayerNorm.layer_norm_eps > 0 else nn.Identity()
        self.fc2 = nn.Linear(mlp_dim, hidden_dim)
        self.dropout2 = nn.Dropout(self.config.Dropout.mlp) if self.config.Dropout.mlp > 0 else nn.Identity()

    def forward(self, x):
        if self.logger:
            self.logger.info("MLP input size is {}".format(x.shape))
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout1(x)
        if self.logger:
            self.logger.info("MLP hidden_layer size is {}".format(x.shape))
        x = self.mlp_norm(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        if self.logger:
            self.logger.info("MLP output size is {}".format(x.shape))
        return x