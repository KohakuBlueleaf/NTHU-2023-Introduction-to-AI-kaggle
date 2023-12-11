from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


def rms_norm(x, scale, eps):
    dtype = reduce(torch.promote_types, (x.dtype, scale.dtype, torch.float32))
    mean_sq = torch.mean(x.to(dtype) ** 2, dim=-1, keepdim=True)
    scale = scale.to(dtype) * torch.rsqrt(mean_sq + eps)
    return x * scale.to(x.dtype)


def normalize(x: torch.Tensor, eps=torch.tensor(1e-4)) -> torch.Tensor:
    """
    modified normalize introduced in https://arxiv.org/abs/2312.02696
    """
    dim = list(range(1, x.ndim))
    n = torch.linalg.vector_norm(x, dim=dim, keepdim=True)
    alpha = torch.sqrt(n.numel() / x.numel())
    return x / torch.add(eps, n, alpha=alpha)


def mp_silu(x, inplace=False):
    """Maginitude-Preserving SiLU, use N(0, 1) as predict sample."""
    return F.silu(x, inplace=inplace) * 1.6766


def mp_mish(x, inplace=False):
    """Maginitude-Preserving Mish, use N(0, 1) as predict sample."""
    return F.mish(x, inplace=inplace) * 1.4868
