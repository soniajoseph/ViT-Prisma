import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, config, logger=None):
        super().__init__()
        self.logger = logger
        self.config = config

        hidden_dim = self.config.transformer.hidden_dim
        mlp_dim = self.config.transformer.mlp_dim

        self.fc1 = nn.Linear(hidden_dim, mlp_dim)
        self.dropout1 = nn.Dropout(self.config.dropout.mlp) if self.config.dropout.mlp > 0 else nn.Identity()
        self.act_fn = self.config.transformer.activation_fn()
        self.mlp_norm = nn.LayerNorm(mlp_dim, eps=self.config.layernorm.layer_norm_eps) if self.config.layernorm.layer_norm_eps > 0 else nn.Identity()
        self.fc2 = nn.Linear(mlp_dim, hidden_dim)
        self.dropout2 = nn.Dropout(self.config.dropout.mlp) if self.config.dropout.mlp > 0 else nn.Identity()

    def _log(self, msg, tensor):
        if self.logger:
            self.logger.info(f"{msg} size: {tensor.shape}")

    def forward(self, x):
        self._log("MLP input", x)

        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout1(x)
        
        self._log("MLP after first FC", x)

        x = self.mlp_norm(x)
        x = self.fc2(x)
        x = self.dropout2(x)

        self._log("MLP output", x)

        return x
