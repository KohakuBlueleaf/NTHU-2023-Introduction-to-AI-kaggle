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
    net = FEClassifierTrainer.load_from_checkpoint(
        r"checkpoints\h16-d0.0-s0.1-w1.0-1.2-attn\epoch=49-step=19500.ckpt"
    )
    net = net.cuda().eval()
    # all_prob = []
    # for _, (inputs, id) in enumerate(tqdm(loader)):
    #     with torch.autocast("cuda", torch.bfloat16):
    #         outputs = net(inputs.cuda())
    #     pred_prob = F.softmax(outputs, dim=1)
    #     for j in range(len(id)):
    #         prob = pred_prob[j][1].item()
    #         all_prob.append(prob)
    # # statstics for all_prob
    # print(
    #     f"mean: {mean(all_prob)}, median: {median(all_prob)}, max: {max(all_prob)}, min: {min(all_prob)}"
    # )

    # threshold = median(all_prob)
    threshold = 0.5
    total1 = 0
    with open("./out/submission.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label"])
        for _, (inputs, id) in enumerate(tqdm(loader)):
            with torch.autocast("cuda", torch.bfloat16):
                outputs = net(inputs.cuda())
            pred_prob = F.softmax(outputs, dim=1)
            predicted = pred_prob[:, 1] >= threshold
            total1 += torch.sum(predicted)
            for j in range(len(id)):
                pred = predicted[j].long().item()
                writer.writerow([id[j].item(), pred])
    print(total1)


if __name__ == "__main__":
    main()
