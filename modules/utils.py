import importlib
import toml
from inspect import isfunction
from random import shuffle
from typing import *

import torch
import torch.nn as nn
from torch.optim import Optimizer

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger


class ProdigyLRMonitor(LearningRateMonitor):
    def _get_optimizer_stats(
        self, optimizer: Optimizer, names: List[str]
    ) -> Dict[str, float]:
        stats = {}
        param_groups = optimizer.param_groups
        use_betas = "betas" in optimizer.defaults

        for pg, name in zip(param_groups, names):
            lr = self._extract_lr(pg, name)
            stats.update(lr)
            momentum = self._extract_momentum(
                param_group=pg,
                name=name.replace(name, f"{name}-momentum"),
                use_betas=use_betas,
            )
            stats.update(momentum)
            weight_decay = self._extract_weight_decay(pg, f"{name}-weight_decay")
            stats.update(weight_decay)

        return stats

    def _extract_lr(self, param_group: Dict[str, Any], name: str) -> Dict[str, Any]:
        lr = param_group["lr"]
        d = param_group.get("d", 1)
        self.lrs[name].append(lr * d)
        return {name: lr * d}


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate(obj):
    if isinstance(obj, str):
        return get_obj_from_str(obj)
    return obj


def exists(val):
    return val is not None


def uniq(arr):
    return {el: True for el in arr}.keys()


def default(val, d):
    if val is not None:
        return val
    return d() if isfunction(d) else d


def zero_module(module: nn.Module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def random_choice(
    x: torch.Tensor,
    num: int,
):
    rand_x = list(x)
    shuffle(rand_x)

    return torch.stack(rand_x[:num])


def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    return total_params
