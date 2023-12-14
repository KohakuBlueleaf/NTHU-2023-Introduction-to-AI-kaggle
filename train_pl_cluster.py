import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from prodigyopt import Prodigy

from modules.trainer import FEClassifierTrainer
from modules.utils import ProdigyLRMonitor
from data.dataset import Dataset
from metadatas import indices, indices_for_clusters

torch.set_float32_matmul_precision("high")


EPOCH = 20
BATCH_SIZE = 2048
KEEP_FEATURES = 16
HIDDEN = 64
DROPOUT = 0.33
SMOOTHING = 0.1
CE_WEIGHT = [1.05, 1.0]
CLUSTER = 4

RMS_NORM = False
USE_ATTN = False
MAGNITUDE_PRESERVING = False


def main():
    pl.seed_everything(0)
    mask = [i for i in range(35) if i in indices_for_clusters[CLUSTER][:KEEP_FEATURES]]
    train_dataset = Dataset(f"cleaned-train-{CLUSTER}", mask)
    train_dataset.balancing()
    valid_dataset = Dataset(f"cleaned-val-{CLUSTER}", mask)

    mu, sigma = torch.load(f"./data/standardize_factor_{CLUSTER}.pth")
    train_dataset.standardize(mu, sigma)
    valid_dataset.standardize(mu, sigma)

    loader = data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )
    val_loader = data.DataLoader(valid_dataset, batch_size=2000)
    total_step = len(loader) * EPOCH

    model = FEClassifierTrainer(
        KEEP_FEATURES,
        num_classes=2,
        hidden=HIDDEN,
        dropout_rate=DROPOUT,
        rms_norm=RMS_NORM,
        use_attn=USE_ATTN,
        maginitude_preserving=MAGNITUDE_PRESERVING,
        loss=nn.CrossEntropyLoss,
        loss_configs={
            "label_smoothing": SMOOTHING,
            "weight": torch.tensor(CE_WEIGHT).float(),
        },
        lr=1.0,
        optimizer=Prodigy,
        lr_scheduler=optim.lr_scheduler.CosineAnnealingLR,
        lr_sch_configs={"T_max": total_step, "eta_min": 1e-2},
        use_warm_up=False,
        shift_to_median=False,
    )
    print(model)

    name = f"{CLUSTER}-h{HIDDEN}-d{DROPOUT}-s{SMOOTHING}-w{CE_WEIGHT[0]}-{CE_WEIGHT[1]}"
    if USE_ATTN:
        name += "-attn"
    if MAGNITUDE_PRESERVING:
        name += "-mp"
    if RMS_NORM:
        name += "-rms"
    save_path = f"./checkpoints/{name}"

    logger = None
    logger = WandbLogger(
        name=name,
        project="AI-kaggle",
        # offline=True,
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        precision="bf16-mixed",
        devices=1,
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
                filename="{epoch}",
                every_n_epochs=1,
            ),
        ],
        num_sanity_val_steps=-1,
    )
    trainer.fit(model, loader, val_loader)


def wandb_sweep_wrapper():
    global EPOCH, BATCH_SIZE, KEEP_FEATURES, HIDDEN, DROPOUT, SMOOTHING, CE_WEIGHT, CLUSTER

    import wandb

    wandb.init(project="AI-kaggle")
    config = wandb.config

    # Apply config's param to global variables
    EPOCH = 20
    BATCH_SIZE = 2048
    KEEP_FEATURES = 24
    CLUSTER = config.cluster

    main()


if __name__ == "__main__":
    main()
    # sweep_config = {
    #     "method": "grid",
    #     "name": "cluster_sweep",
    #     "parameters": {
    #         "cluster": {"values": [0, 1, 2, 3, 4]},
    #     },
    # }

    # import wandb

    # sweep_id = wandb.sweep(sweep_config, project="AI-kaggle")
    # wandb.agent(sweep_id, function=wandb_sweep_wrapper)
