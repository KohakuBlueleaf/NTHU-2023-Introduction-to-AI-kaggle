from typing import *

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sch
import pytorch_lightning as pl
from warmup_scheduler import GradualWarmupScheduler

from modules.utils import instantiate
from modules.model import FEClassifier


class BaseTrainer(pl.LightningModule):
    def __init__(
        self,
        *args,
        name="",
        lr: float = 1e-5,
        optimizer: type[optim.Optimizer] = optim.AdamW,
        opt_configs: dict[str, Any] = {
            "weight_decay": 0.01,
            "betas": (0.9, 0.999),
        },
        lr_scheduler: Optional[type[lr_sch.LRScheduler]] = lr_sch.CosineAnnealingLR,
        lr_sch_configs: dict[str, Any] = {
            "T_max": 100_000,
            "eta_min": 1e-7,
        },
        use_warm_up: bool = True,
        warm_up_period: int = 1000,
        **kwargs,
    ):
        super(BaseTrainer, self).__init__()
        self.name = name
        self.train_params: Iterator[nn.Parameter] = None
        self.optimizer = instantiate(optimizer)
        self.opt_configs = opt_configs
        self.lr = lr
        self.lr_sch = instantiate(lr_scheduler)
        self.lr_sch_configs = lr_sch_configs
        self.use_warm_up = use_warm_up
        self.warm_up_period = warm_up_period

    def configure_optimizers(self):
        assert self.train_params is not None
        optimizer = self.optimizer(self.train_params, lr=self.lr, **self.opt_configs)

        lr_sch = None
        if self.lr_sch is not None:
            lr_sch = self.lr_sch(optimizer, **self.lr_sch_configs)

        if self.use_warm_up:
            lr_scheduler = GradualWarmupScheduler(
                optimizer, 1, self.warm_up_period, lr_sch
            )
        else:
            lr_scheduler = lr_sch

        if lr_scheduler is None:
            return optimizer
        else:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": lr_scheduler, "interval": "step"},
            }


class FEClassifierTrainer(FEClassifier, BaseTrainer):
    def __init__(
        self,
        in_features=35,
        num_classes=2,
        hidden=512,
        dropout_rate=0.0,
        loss=nn.CrossEntropyLoss,
        loss_configs: dict[str, Any] = {"weight": None, "label_smoothing": 0.0},
        name="",
        lr: float = 1e-5,
        optimizer: type[optim.Optimizer] = optim.AdamW,
        opt_configs: dict[str, Any] = {},
        lr_scheduler: Optional[type[lr_sch.LRScheduler]] = lr_sch.CosineAnnealingLR,
        lr_sch_configs: dict[str, Any] = {},
        use_warm_up: bool = False,
        warm_up_period: int = 0,
        shift_to_median: bool = False,
    ):
        super(FEClassifierTrainer, self).__init__(
            in_features=in_features,
            num_classes=num_classes,
            hidden=hidden,
            dropout_rate=dropout_rate,
            loss=loss,
            loss_configs=loss_configs,
            name=name,
            lr=lr,
            optimizer=optimizer,
            opt_configs=opt_configs,
            lr_scheduler=lr_scheduler,
            lr_sch_configs=lr_sch_configs,
            use_warm_up=use_warm_up,
            warm_up_period=warm_up_period,
        )
        self.save_hyperparameters()
        self.train_params = self.parameters()
        self.loss = instantiate(loss)(**loss_configs)
        self.loss_configs = loss_configs
        self.shift_to_median = shift_to_median
        self.reset_state()

    def reset_state(self):
        self.total = 0
        self.acc = 0
        self.true_positive = 0
        self.true_negative = 0
        self.false_positive = 0
        self.false_negative = 0
        self.f1_score = 0
        self.precision = 0
        self.recall = 0

    def on_train_epoch_start(self) -> None:
        self.reset_state()

    def training_step(self, batch, idx):
        self.train()
        inputs, labels = batch

        outputs = self(inputs)
        predicted = torch.argmax(outputs.data, 1)
        loss = self.loss(outputs.float(), labels)

        self.total += labels.size(0)
        self.true_positive += ((predicted == 1) & (labels == 1)).sum().item()
        self.true_negative += ((predicted == 0) & (labels == 0)).sum().item()
        self.false_positive += ((predicted == 1) & (labels == 0)).sum().item()
        self.false_negative += ((predicted == 0) & (labels == 1)).sum().item()

        correct = self.true_positive + self.true_negative
        self.acc = correct / self.total
        self.precision = self.true_positive / max(
            self.true_positive + self.false_positive, 1
        )
        self.recall = self.true_positive / max(
            self.true_positive + self.false_negative, 1
        )
        self.f1_score = (
            2 * self.precision * self.recall / max(self.precision + self.recall, 1e-8)
        )

        if self._trainer is not None:
            self.log(
                "train/loss", float(loss), on_step=True, logger=True, prog_bar=True
            )
            self.log("train/acc", float(self.acc), on_step=True, logger=True)
            self.log("train/f1_score", float(self.f1_score), on_step=True, logger=True)
            self.log(
                "train/precision", float(self.precision), on_step=True, logger=True
            )
            self.log("train/recall", float(self.recall), on_step=True, logger=True)

        return loss

    def on_validation_epoch_start(self) -> None:
        self.probs = []
        self.targets = []

    @torch.no_grad()
    def validation_step(self, batch, idx):
        self.eval()
        inputs, labels = batch

        outputs = self(inputs)
        self.probs.append(F.softmax(outputs, dim=1).cpu())
        self.targets.append(labels.cpu())

    def on_validation_epoch_end(self) -> None:
        probs = torch.concat(self.probs).numpy()
        targets = torch.concat(self.targets).numpy()
        if self.shift_to_median:
            median_of_prob = np.median(probs[:, 1], axis=0)
            pred = probs[:, 1] >= median_of_prob
        else:
            pred = probs.argmax(axis=1)

        true_pos = ((targets == 1) & pred).sum()
        true_neg = ((targets == 0) & ~pred).sum()
        false_pos = ((targets == 0) & pred).sum()
        false_neg = ((targets == 1) & ~pred).sum()

        total = pred.shape[0]
        correct = true_pos + true_neg
        acc = correct / total
        precision = true_pos / max(true_pos + false_pos, 1)
        recall = true_pos / max(true_pos + false_neg, 1)
        f1_score = 2 * precision * recall / max(precision + recall, 1e-8)
        f1_score = f1_score

        if self._trainer is not None:
            self.log("val/acc", float(acc), on_step=False, logger=True)
            self.log("val/f1_score", float(f1_score), on_step=False, logger=True)
            self.log("val/precision", float(precision), on_step=False, logger=True)
            self.log("val/recall", float(recall), on_step=False, logger=True)
