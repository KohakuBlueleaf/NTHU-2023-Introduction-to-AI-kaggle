import sys
from multiprocessing import Pool

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


EPOCH = 75
BATCH_SIZE = 2048
KEEP_FEATURES = 24
HIDDEN = 128
DROPOUT = 0.5
SMOOTHING = 0.1
CE_WEIGHT = [1.1, 1.0]
D_COEF = 0.8


RMS_NORM = False
USE_ATTN = False
MAGNITUDE_PRESERVING = False


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
        opt_configs={"d_coef": D_COEF},
        lr_scheduler=optim.lr_scheduler.CosineAnnealingLR,
        lr_sch_configs={"T_max": total_step, "eta_min": 1e-2},
        use_warm_up=False,
        shift_to_median=False,
    )
    print(model)

    name = f"EP{EPOCH}-B{BATCH_SIZE}-h{HIDDEN}-d{DROPOUT}-s{SMOOTHING}-w{CE_WEIGHT[0]}-{CE_WEIGHT[1]}"
    if USE_ATTN:
        name += "-attn"
    if MAGNITUDE_PRESERVING:
        name += "-mp"
    if RMS_NORM:
        name += "-rms"
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
                filename="epoch={epoch}-f1_score={val/f1_score:.5f}-acc={val/acc:.5f}",
                save_last=True,
                every_n_epochs=1,
                auto_insert_metric_name=False,
            ),
        ],
        num_sanity_val_steps=-1,
    )
    trainer.fit(model, loader, val_loader)


def wandb_sweep_wrapper():
    global EPOCH, BATCH_SIZE, KEEP_FEATURES, HIDDEN, DROPOUT, SMOOTHING, CE_WEIGHT

    import wandb

    wandb.init(project="AI-kaggle")
    config = wandb.config

    # Apply config's param to global variables
    EPOCH = 50
    BATCH_SIZE = 2048
    KEEP_FEATURES = 24
    HIDDEN = getattr(config, "hidden", None) or HIDDEN
    DROPOUT = getattr(config, "dropout", None) or DROPOUT
    SMOOTHING = getattr(config, "smooth", None) or SMOOTHING
    CE_WEIGHT = getattr(config, "weight", None) or CE_WEIGHT

    main(
        getattr(
            config,
            "seed",
        )
    )


if __name__ == "__main__":
    mp_pool = Pool(15)
    for seed in range(15):
        mp_pool.apply_async(main, args=(seed + 3407,))
    mp_pool.close()
    mp_pool.join()
    # main(0)
    # sweep_config = {
    #     "method": "grid",
    #     "name": "first_sweep",
    #     "metric": {
    #         "goal": "minimize",
    #         "name": "val/f1_score",
    #     },
    #     "parameters": {
    #         "seed": {"values": list(range(5))},
    #     },
    # }

    # import wandb

    # sweep_id = wandb.sweep(sweep_config, project="AI-kaggle")
    # wandb.agent(sweep_id, function=wandb_sweep_wrapper)
