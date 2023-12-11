import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import (
    MPMish,
    MPSiLU,
    ForcedWeightNormLinear,
    FeatureSelfAttention,
    RMSNorm,
)


class FeatureExpander(nn.Module):
    def __init__(
        self,
        in_features=35,
        hidden=128,
        dropout_rate=0.0,
        rms_norm=False,
        use_attn=False,
        maginitude_preserving=False,
    ):
        super(FeatureExpander, self).__init__()

        if maginitude_preserving:
            linear_cls = ForcedWeightNormLinear
            act = MPMish
        else:
            linear_cls = nn.Linear
            act = nn.Mish

        if rms_norm:
            norm_cls = RMSNorm
        else:
            norm_cls = nn.LayerNorm

        self.input_proj = nn.Sequential(
            linear_cls(in_features, hidden),
            nn.Identity() if maginitude_preserving else norm_cls(hidden),
        )
        self.mlp = nn.Sequential(
            nn.Dropout(dropout_rate),
            linear_cls(hidden, hidden * 4),
            act(),
            linear_cls(hidden * 4, hidden),
            nn.Dropout(dropout_rate),
        )
        self.ws = nn.Parameter(torch.ones(1, hidden))

        self.use_attn = use_attn
        if use_attn:
            self.use_attn = use_attn
            self.norm = nn.Identity() if maginitude_preserving else norm_cls(hidden)
            self.attn = FeatureSelfAttention(hidden=hidden, dropout=dropout_rate)
            self.ws_attn = nn.Parameter(torch.ones(1, hidden))

    def forward(self, x):
        x = self.input_proj(x)
        if self.use_attn:
            h = self.attn(x)
            h = self.ws_attn * h + (1 - self.ws_attn) * h
            x = self.norm(h)
        h = self.mlp(x)
        h = self.ws * h + (1 - self.ws) * x
        return h


class FEClassifier(nn.Module):
    def __init__(
        self,
        in_features=35,
        num_classes=2,
        hidden=128,
        dropout_rate=0.0,
        rms_norm=False,
        use_attn=False,
        maginitude_preserving=False,
        **kwargs
    ):
        super(FEClassifier, self).__init__(**kwargs)
        if maginitude_preserving:
            linear_cls = ForcedWeightNormLinear
        else:
            linear_cls = nn.Linear

        self.expander = FeatureExpander(
            in_features=in_features,
            hidden=hidden,
            dropout_rate=dropout_rate,
            rms_norm=rms_norm,
            use_attn=use_attn,
            maginitude_preserving=maginitude_preserving,
        )
        self.head = linear_cls(hidden, num_classes)

    def forward(self, x):
        h = self.expander(x)
        h = self.head(h)
        return h
