import csv
import pickle
import random

import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, datasets, decomposition
from tqdm import trange

from dataset import Dataset

# dataset = Dataset('cleaned-full')
dataset2 = Dataset("cleaned-val")

# Prepare the data
# x = np.array(dataset.all_data)
# y = np.array(dataset.label)
# n_samples, n_features = x.shape

x2 = np.array(random.choices(dataset2.all_data, k=10000))
n_samples2, n_features2 = x2.shape
y2 = np.ones(n_samples2) * 2

# t-SNE
pca: decomposition.PCA
pca, x_min, x_max = pickle.load(open("./models/pca.pkl", "rb"))
# X_tsne = manifold.TSNE(n_components=2, init='random', random_state=5, verbose=1).fit_transform(x)

# x_val = pca.transform(x)
x2_val = pca.transform(x2)

# Data Visualization
# x_min, x_max = x_val.min(0), x_val.max(0)
# print(x_min, x_max)
# X_norm = (x_val - x_min) / (x_max - x_min)  #Normalize
x2_norm = (x2_val - x_min) / (x_max - x_min)
# with open('./models/pca.pkl', 'wb') as f:
#     pickle.dump([pca, x_min, x_max], f)
plt.figure(figsize=(8, 8))
good_samples = []
# for i in trange(X_norm.shape[0]):
#     plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]),
#              fontdict={'weight': 'bold', 'size': 9})
for i in trange(x2_norm.shape[0]):
    plt.text(
        x2_norm[i, 0],
        x2_norm[i, 1],
        "2",
        color=plt.cm.Set1(y2[i]),
        fontdict={"weight": "bold", "size": 9},
    )
plt.xticks([])
plt.yticks([])
plt.show()
