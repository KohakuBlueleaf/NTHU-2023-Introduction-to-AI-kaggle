import csv
from random import shuffle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

from tqdm import tqdm


class Dataset(data.Dataset):
    def __init__(self, split="train", mask=None):
        # read csv from f"{split}.csv"
        self.all_data = []
        self.data0 = []
        self.data1 = []
        self.label = []
        with open(f"data/{split}.csv") as f:
            reader = csv.reader(f)
            next(reader)
            for data in reader:
                self.label.append(int(data.pop(0)))
                self.all_data.append([float(x) for x in data])
                if self.label[-1]:
                    self.data1.append([float(x) for x in data])
                else:
                    self.data0.append([float(x) for x in data])
        print(len(self.all_data), len(self.data0), len(self.data1))
        self.datas = torch.tensor(self.all_data)
        self.targets = torch.tensor(self.label)
        self.mask = mask or list(range(len(self.all_data[0])))

    def standardize(self, mu, sigma):
        mu = mu.unsqueeze(0)
        sigma = sigma.unsqueeze(0)
        self.datas = (self.datas - mu) / sigma

    def normalize(self, mu, min, max):
        mu = mu.unsqueeze(0)
        min = min.unsqueeze(0)
        max = max.unsqueeze(0)
        self.datas = (self.datas - min) / (max - min)

    def balancing(self, scale=1):
        if len(self.data0) > len(self.data1):
            self.data1 = self.data1 * max(
                (len(self.data0) // len(self.data1) + 1), scale
            )
        else:
            self.data0 = self.data0 * max(
                (len(self.data1) // len(self.data0) + 1), scale
            )
        self.datas = torch.tensor(self.data0 + self.data1)
        self.targets = torch.concat(
            [
                torch.zeros(len(self.data0), dtype=torch.long),
                torch.ones(len(self.data1), dtype=torch.long),
            ]
        )

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        return self.datas[idx][self.mask], self.targets[idx]


if __name__ == "__main__":
    # Get the mean and standard deviation
    for i in range(5):
        dataset = Dataset(f"cleaned-full-{i}")
        mu = torch.mean(dataset.datas, dim=0)
        min = torch.min(dataset.datas, dim=0)
        max = torch.max(dataset.datas, dim=0)
        sigma = torch.std(dataset.datas, dim=0)

        print(mu, sigma)

        torch.save([mu, min, max], f"./data/normalize_factor_{i}.pth")
        torch.save([mu, sigma], f"./data/standardize_factor_{i}.pth")
