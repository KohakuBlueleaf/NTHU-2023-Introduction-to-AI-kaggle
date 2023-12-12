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
from metadatas import indices

torch.set_float32_matmul_precision("high")


EPOCH = 50
BATCH_SIZE = 2048
KEEP_FEATURES = 24
HIDDEN = 16
DROPOUT = 0.0
SMOOTHING = 0.1
CE_WEIGHT = [1.0, 1.0]

RMS_NORM = False
USE_ATTN = True
MAGNITUDE_PRESERVING = False


def main():
    pl.seed_everything(0)
    mask = [i for i in range(35) if i in indices[:KEEP_FEATURES]]
    train_dataset = Dataset("cleaned-train", mask)
    train_dataset.balancing()
    valid_dataset = Dataset("cleaned-val", mask)

    mu, sigma = torch.load("./data/standardize_factor.pth")
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

    name = (
        f"h{HIDDEN}-d{DROPOUT}-s{SMOOTHING}-w{CE_WEIGHT[0]}-{CE_WEIGHT[1]}"
    )
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
        log_model="all",
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
            ModelCheckpoint(dirpath=save_path, every_n_epochs=1),
        ],
        num_sanity_val_steps=-1,
    )
    trainer.fit(model, loader, val_loader)


def wandb_sweep_wrapper():
    global EPOCH, BATCH_SIZE, KEEP_FEATURES, HIDDEN, DROPOUT, SMOOTHING, CE_WEIGHT

    import wandb

    wandb.init(project="AI-kaggle-sweep")
    config = wandb.config

    # Apply config's param to global variables
    EPOCH = 50
    BATCH_SIZE = 2048
    KEEP_FEATURES = 24
    HIDDEN = config.hidden
    DROPOUT = config.dropout
    SMOOTHING = config.smooth
    CE_WEIGHT = config.weight

    main()


if __name__ == "__main__":
    main()
    # sweep_config = {
    #     "method": "grid",
    #     "name": "first_sweep",
    #     "metric": {
    #         "goal": "minimize",
    #         "name": "val/f1_score",
    #     },
    #     "parameters": {
    #         "hidden": {"values": [128, 192, 256]},
    #         "dropout": {"values": [0.25, 0.5, 0.75]},
    #         "smooth": {"values": [0.0, 0.1, 0.2]},
    #         "weight": {
    #             "values": [[1.0, 1.0], [1.0, 1.2], [1.2, 1.0], [1.0, 1.05], [1.05, 1.0]]
    #         },
    #     },
    # }

    # import wandb

    # sweep_id = wandb.sweep(sweep_config, project="AI-kaggle-sweep")
    # wandb.agent(sweep_id, function=wandb_sweep_wrapper)
