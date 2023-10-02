import torch.nn as nn

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(self.config.hidden_dim, self.config.mlp_dim)
        self.dropout1 = nn.Dropout(self.config.mlp_dropout) if self.config.mlp_dropout > 0 else nn.Identity()
        self.act_fn = self.config.activation_fn()
        self.norm = nn.LayerNorm(self.config.hidden_dim, eps=self.config.layer_norm_eps) if self.config.layer_norm_eps > 0 else nn.Identity()
        self.fc2 = nn.Linear(self.config.mlp_dim, self.config.hidden_dim)
        self.dropout2 = nn.Dropout(self.config.mlp_dropout) if self.config.mlp_dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x