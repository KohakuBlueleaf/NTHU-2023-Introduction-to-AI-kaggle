# load 'data/data.npy' and 'data/label.npy' and visualize them via PCA
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from data.dataset import Dataset


mu, sigma = torch.load("./data/standardize_factor.pth")
mu, x_min, x_max = torch.load("./data/normalize_factor.pth")
dataset = Dataset("cleaned-full")

# dataset.standardize(mu, sigma)


data = dataset.datas.numpy()
label = dataset.targets.numpy()


indices = [
    2,
    1,
    9,
    19,
    21,
    3,
    12,
    13,
    22,
    33,
    6,
    18,
    24,
    5,
    26,
    8,
    4,
    28,
    27,
    11,
    0,
    16,
    30,
    15,
    31,
    32,
    34,
    29,
    14,
    17,
    23,
    25,
    7,
    20,
    10,
]
keep_k = 9

print("Keep", keep_k, "features at index", indices[:keep_k])

data = data[:, indices[:keep_k]]
# do 3d PCA
pca = PCA(n_components=3)
pca.fit(data)
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)
data = pca.transform(data)

# plot
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
# make the figure bigger
fig.set_size_inches(10, 10)
ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=label, s=1)
plt.show()
plt.savefig("visualize/pca2.png")
