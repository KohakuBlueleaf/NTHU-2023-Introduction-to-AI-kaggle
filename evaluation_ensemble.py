import csv
from statistics import mean, median

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

from tqdm import tqdm

from data.dataset import Dataset
from modules.trainer import FEClassifierTrainer
from metadatas import indices

torch.set_float32_matmul_precision("high")


KEEP_FEATURES = 24


@torch.no_grad()
def main():
    mask = [i for i in range(35) if i in indices[:KEEP_FEATURES]]
    dataset = Dataset("test", mask)
    dataset.standardize(*torch.load("./data/standardize_factor.pth"))

    loader = data.DataLoader(dataset, batch_size=1000)
    nets = [
        FEClassifierTrainer.load_from_checkpoint(
            f"checkpoints\\EP75-B2048-h128-d0.5-s0.1-w1.1-1.0-dcoef0.8-seed{i+3407}\\last.ckpt"
        ).cuda().eval()
        for i in range(15)
    ]
    total1 = 0
    with open("./out/submission.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label"])
        for _, (inputs, id) in enumerate(tqdm(loader)):
            pred_prob = 0
            for net in nets:
                with torch.autocast("cuda", torch.bfloat16):
                    outputs = net(inputs.cuda())
                pred_prob += F.softmax(outputs, dim=1)
            predicted = pred_prob.argmax(dim=1)
            total1 += torch.sum(predicted)
            for j in range(len(id)):
                pred = predicted[j].long().item()
                writer.writerow([id[j].item(), pred])
    print(total1)


if __name__ == "__main__":
    main()
