import csv
import os

import torch
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
    net = FEClassifierTrainer.load_from_checkpoint(
        r"checkpoints/EP100-B2048-h128-d0.5-s0.1-w1.1-1.0-dcoef0.75-seed3416/last.ckpt"
    )
    net = net.cuda().eval()
    total1 = 0
    os.makedirs("./out", exist_ok=True)
    with open("./out/submission.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label"])
        for _, (inputs, id) in enumerate(tqdm(loader)):
            with torch.autocast("cuda", torch.bfloat16):
                outputs = net(inputs.cuda())
            predicted = outputs.argmax(dim=1)
            total1 += torch.sum(predicted)
            for j in range(len(id)):
                pred = predicted[j].long().item()
                writer.writerow([id[j].item(), pred])
    print(total1)


if __name__ == "__main__":
    main()
