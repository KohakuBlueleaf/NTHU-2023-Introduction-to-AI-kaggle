import csv
import os


ranges = [
    [546.10, 579.18],
    [843.76, 876.84],
    [1108.35, 1141.43],
    [1406.01, 1439.09],
    [1670.60, 1703.68],
]
spliters = [(j[0] + i[1]) / 2 for i, j in zip(ranges[:-1], ranges[1:])]


def split(value):
    for idx, spliter in enumerate(spliters):
        if value < spliter:
            return idx
    return len(spliters)


def split_csv(file):
    basename = os.path.splitext(os.path.basename(file))[0]
    for i in range(5):
        with open(f"{basename}-{i}.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["label", *range(35)])
    with open(file) as f:
        reader = csv.reader(f)
        next(reader)
        for data in reader:
            idx = split(float(data[20]))
            with open(f"{basename}-{idx}.csv", "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(data)


if __name__ == "__main__":
    split_csv("cleaned-full.csv")
