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
from metadatas import indices, indices_for_clusters

torch.set_float32_matmul_precision("high")


KEEP_FEATURES = 24


@torch.no_grad()
def main():
    for cluster in range(5):
        mask = [
            i for i in range(35) if i in indices_for_clusters[cluster][:KEEP_FEATURES]
        ]
        dataset = Dataset(f"test-{cluster}", mask)
        dataset.standardize(*torch.load(f"./data/standardize_factor_{cluster}.pth"))

        loader = data.DataLoader(dataset, batch_size=1000)
        try:
            net = FEClassifierTrainer.load_from_checkpoint(
                f"checkpoints\\{cluster}-h64-d0.25-s0.1-w1.05-1.0\\epoch=19-step=1540.ckpt"
            )
        except:
            net = FEClassifierTrainer.load_from_checkpoint(
                f"checkpoints\\{cluster}-h64-d0.25-s0.1-w1.05-1.0\\epoch=19-step=1560.ckpt"
            )
        net = net.cuda().eval()
        total1 = 0
        with open(f"./out/submission{cluster}.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "label"])
            for _, (inputs, idx) in enumerate(tqdm(loader)):
                with torch.autocast("cuda", torch.bfloat16):
                    outputs = net(inputs.to(net.device))
                predicted = outputs.argmax(dim=1)
                total1 += torch.sum(predicted)
                for j, id in enumerate(idx):
                    pred = predicted[j].long().item()
                    writer.writerow([id.item(), pred])
        print(total1)


if __name__ == "__main__":
    main()

    # Merge all submissions
    datas = []
    for i in range(5):
        with open(f"out/submission{i}.csv") as f2:
            reader = csv.reader(f2)
            next(reader)
            for d in reader:
                datas.append((int(d[0]), int(d[1])))
    datas.sort()
    with open("out/submission.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label"])
        for d in datas:
            writer.writerow([d[0], d[1]])
