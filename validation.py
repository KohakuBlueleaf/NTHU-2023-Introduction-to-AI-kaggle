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
    dataset = Dataset("cleaned-val", mask)
    dataset.standardize(*torch.load("./data/standardize_factor.pth"))

    loader = data.DataLoader(dataset, batch_size=1000)
    net = FEClassifierTrainer.load_from_checkpoint(
        r"checkpoints/h16-d0.0-s0.0-w1.0-1.0-attn-rms/epoch=38-f1_score=0.87801.ckpt"
    )
    net = net.eval()
    all_prob = []
    for _, (inputs, id) in enumerate(tqdm(loader)):
        with torch.autocast("cuda", torch.bfloat16):
            outputs = net(inputs.to(net.device))
        pred_prob = F.softmax(outputs, dim=1).cpu()
        for j in range(len(id)):
            prob = pred_prob[j][1].item()
            all_prob.append(prob)
    # statstics for all_prob
    print(
        f"mean: {mean(all_prob)}, median: {median(all_prob)}, max: {max(all_prob)}, min: {min(all_prob)}"
    )

    total = 0
    correct = 0
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    threshold = median(all_prob)
    threshold = 0.5
    for _, (inputs, id) in enumerate(tqdm(loader)):
        with torch.autocast("cuda", torch.bfloat16):
            outputs = net(inputs.to(net.device))
        pred_prob = F.softmax(outputs, dim=1).cpu()
        predicted = pred_prob[:, 1] >= threshold
        true_positive += (predicted & (id == 1)).sum().item()
        true_negative += (~predicted & (id == 0)).sum().item()
        false_positive += (predicted & (id == 0)).sum().item()
        false_negative += (~predicted & (id == 1)).sum().item()
        correct += (predicted == id).sum().item()
        total += predicted.size(0)
    correct = true_positive + true_negative
    acc = correct / total
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1_score = 2 * precision * recall / max(precision + recall, 1e-8)

    print(f"Validation Accuracy: {acc*100:.3f} %")
    print(f"Validation Precision: {precision*100:.3f} %")
    print(f"Validation Recall: {recall*100:.3f} %")
    print(f"Validation F1 Score: {f1_score:.5f}")


if __name__ == "__main__":
    main()
