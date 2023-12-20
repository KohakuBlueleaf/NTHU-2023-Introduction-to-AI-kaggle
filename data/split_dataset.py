import csv
from random import shuffle
from .dataset import Dataset


full_dataset = Dataset("cleaned-full")
data0 = full_dataset.data0
data1 = full_dataset.data1
shuffle(data0)
shuffle(data1)

# Split balanced val set
train_data0 = data0[:5000]
train_data1 = data1[:5000]
val_data0 = data0[5000:]
val_data1 = data1[5000:]

with open("data/cleaned-train.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["label", *range(35)])
    for data in train_data0:
        writer.writerow([0, *data])
    for data in train_data1:
        writer.writerow([1, *data])

with open("data/cleaned-val.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["label", *range(35)])
    for data in val_data0:
        writer.writerow([0, *data])
    for data in val_data1:
        writer.writerow([1, *data])
