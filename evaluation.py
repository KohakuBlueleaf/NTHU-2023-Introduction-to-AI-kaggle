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
    """
    Runs the main function.

    This function performs the following steps:
    1. Creates a mask based on the indices.
    2. Initializes a Dataset object with the mask.
    3. Loads the standardization factors from the "./data/standardize_factor.pth" file.
    4. Creates a DataLoader object with the dataset and a batch size of 1000.
    5. Loads a pre-trained FEClassifierTrainer model from a checkpoint file.
    6. Moves the model to the GPU and sets it to evaluation mode.
    7. Sets the threshold for classification to 0.5.
    8. Opens the "./out/submission.csv" file for writing.
    9. Writes the header row to the CSV file.
    10. Iterates over the DataLoader and performs the following steps for each batch:
        a. Performs forward pass on the model with the batch inputs.
        b. Applies softmax activation to the model outputs.
        c. Classifies the predictions based on the threshold.
        d. Updates the total count of positive predictions.
        e. Writes the batch predictions to the CSV file.
    11. Prints the total count of positive predictions.
    """
    mask = [i for i in range(35) if i in indices[:KEEP_FEATURES]]
    dataset = Dataset("test", mask)
    dataset.standardize(*torch.load("./data/standardize_factor.pth"))

    loader = data.DataLoader(dataset, batch_size=1000)
    net = FEClassifierTrainer.load_from_checkpoint(
        r"checkpoints\h128-d0.5-s0.1-w1.05-1.0-seed11\epoch=49-step=19500.ckpt"
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
