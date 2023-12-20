import torch
import torch.nn as nn
import torch.nn.functional as F


act = nn.Mish


class FeatureExpander(nn.Module):
    def __init__(
        self,
        in_features=35,
        hidden=128,
        dropout_rate=0.0,
    ):
        super(FeatureExpander, self).__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.LayerNorm(hidden),
        )
        self.mlp = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden, hidden * 4),
            act(),
            nn.Linear(hidden * 4, hidden),
            nn.Dropout(dropout_rate),
        )
        self.ws = nn.Parameter(torch.ones(1, hidden))

    def forward(self, x):
        x = self.input_proj(x)
        h = self.mlp(x)
        h = self.ws * h + (1 - self.ws) * x
        return h


class FEClassifier(nn.Module):
    def __init__(
        self, in_features=35, num_classes=2, hidden=128, dropout_rate=0.0, **kwargs
    ):
        super(FEClassifier, self).__init__(**kwargs)

        self.expander = FeatureExpander(
            in_features=in_features,
            hidden=hidden,
            dropout_rate=dropout_rate,
        )
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x):
        h = self.expander(x)
        h = self.head(h)
        return h
