import pickle
import csv

import numpy as np
from sklearn import decomposition
from tqdm import trange

from dataset import Dataset

dataset = Dataset("train")

# Prepare the data
x = np.array(dataset.all_data)
y = dataset.label
n_samples, n_features = x.shape


pca = decomposition.PCA(n_components=2).fit(x)
x_val = pca.transform(x)

x_min, x_max = x_val.min(0), x_val.max(0)
print(x_min, x_max)
x_norm = (x_val - x_min) / (x_max - x_min)  # Normalize
with open("../models/pca.pkl", "wb") as f:
    pickle.dump([pca, x_min, x_max], f)

good_samples = []
for i in trange(x_norm.shape[0]):
    if x_norm[i][0] > 0.8 and x_norm[i][1] > 0.8 and y[i] == 0:
        continue
    good_samples.append([y[i], *[float(v) for v in x[i]]])


with open("./cleaned-full.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["label"] + [str(i) for i in range(n_features)])
    writer.writerows(good_samples)
