import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from .functions import *


class MPMish(nn.Mish):
    def forward(self, x):
        return super().forward(x) * 1.4868


class MPSiLU(nn.SiLU):
    def forward(self, x):
        return super().forward(x) * 1.6766


class RMSNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return rms_norm(x, self.weight, self.eps) + self.bias


class ForcedWeightNormLinear(nn.Linear):
    """
    Forced Weight Norm for Maginitude-Preserving
    Introduced in https://arxiv.org/abs/2312.02696
    """

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, std=1.0)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    @property
    def normed_weight(self):
        if self.training:
            with torch.no_grad():
                self.weight.copy_(normalize(self.weight))
        fan_in = self.weight[0].numel()
        if not self.training:
            w = normalize(self.weight) / np.sqrt(fan_in)
        else:
            w = self.weight / np.sqrt(fan_in)
        return w

    def forward(self, x):
        return F.linear(x, self.normed_weight, self.bias)


class FeatureSelfAttention(nn.Module):
    def __init__(self, hidden=128, dropout=0.0, linear_cls=nn.Linear):
        super().__init__()
        self.q = linear_cls(hidden, hidden, bias=False)
        self.k = linear_cls(hidden, hidden, bias=False)
        self.v = linear_cls(hidden, hidden, bias=False)
        self.o = linear_cls(hidden, hidden)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        attn_scores = torch.einsum("bm,bn->bmn", q, k)
        attn_scores = F.softmax(attn_scores, dim=-1)
        attn_scores = self.drop(attn_scores)
        attn_out = torch.einsum("bmn,bn->bm", attn_scores, v)

        return self.o(attn_out)
