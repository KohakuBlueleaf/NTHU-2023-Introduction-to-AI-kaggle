import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from prodigyopt import Prodigy

from modules.trainer import FEClassifierTrainer
from modules.utils import ProdigyLRMonitor
from data.dataset import Dataset
from metadatas import indices

torch.set_float32_matmul_precision("high")


EPOCH = 100
BATCH_SIZE = 2048
KEEP_FEATURES = 24
HIDDEN = 128
DROPOUT = 0.5
SMOOTHING = 0.1
CE_WEIGHT = [1.1, 1.0]
D_COEF = 0.8
GPUS = 2


def main(seed=0):
    pl.seed_everything(seed)
    mask = [i for i in range(35) if i in indices[:KEEP_FEATURES]]
    train_dataset = Dataset("cleaned-train", mask)
    train_dataset.balancing()
    valid_dataset = Dataset("cleaned-val", mask)

    mu, sigma = torch.load("./data/standardize_factor.pth")
    train_dataset.standardize(mu, sigma)
    valid_dataset.standardize(mu, sigma)

    loader = data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
        num_workers = 4,
        persistent_workers = 4,
    )
    val_loader = data.DataLoader(valid_dataset, batch_size=2000,
        num_workers = 2,
        persistent_workers = 2,
    )
    total_step = len(loader) * EPOCH

    model = FEClassifierTrainer(
        KEEP_FEATURES,
        num_classes=2,
        hidden=HIDDEN,
        dropout_rate=DROPOUT,
        loss=nn.CrossEntropyLoss,
        loss_configs={
            "label_smoothing": SMOOTHING,
            "weight": torch.tensor(CE_WEIGHT).float(),
        },
        lr=1.0,
        optimizer=Prodigy,
        opt_configs={"d_coef": D_COEF},
        lr_scheduler=optim.lr_scheduler.CosineAnnealingLR,
        lr_sch_configs={"T_max": total_step, "eta_min": 1e-2},
        use_warm_up=False,
        shift_to_median=False,
    )
    print(model)

    name = f"EP{EPOCH}-B{BATCH_SIZE}-h{HIDDEN}-d{DROPOUT}-s{SMOOTHING}-w{CE_WEIGHT[0]}-{CE_WEIGHT[1]}"
    if D_COEF != 1.0:
        name += f"-dcoef{D_COEF}"
    name += f'-seed{seed}'
    save_path = f"./checkpoints/{name}"

    logger = None
    logger = WandbLogger(
        name=name,
        project="AI-kaggle",
        # offline=True,
    )
    print(seed % GPUS)
    trainer = pl.Trainer(
        accelerator="gpu",
        precision="bf16-mixed",
        devices=[seed % GPUS],
        max_epochs=EPOCH,
        logger=logger,
        log_every_n_steps=1,
        callbacks=[
            ProdigyLRMonitor(logging_interval="step"),
            ModelCheckpoint(
                dirpath=save_path,
                filename="epoch={epoch}-f1_score={val/f1_score:.5f}",
                monitor="val/f1_score",
                mode="max",
                every_n_epochs=1,
                save_top_k=2,
                auto_insert_metric_name=False,
            ),
            ModelCheckpoint(
                dirpath=save_path,
                filename="epoch={epoch}-f1_score={val/f1_score:.5f}-acc={val/acc:.5f}",
                save_last=True,
                every_n_epochs=1,
                auto_insert_metric_name=False,
            ),
        ],
        num_sanity_val_steps=-1,
    )
    trainer.fit(model, loader, val_loader)


if __name__ == "__main__":
    seed = 3407
    if len(sys.argv) > 1:
        seed = int(sys.argv[1])
    main(seed)
