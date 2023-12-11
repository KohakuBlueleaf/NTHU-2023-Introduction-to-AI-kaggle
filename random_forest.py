import torch

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from data.dataset import Dataset


mu, sigma = torch.load("./standardize_factor.pth")
mu, x_min, x_max = torch.load("./normalize_factor.pth")
train_dataset = Dataset("cleaned-train")
val_dataset = Dataset("cleaned-val")

train_dataset.standardize(mu, sigma)
val_dataset.standardize(mu, sigma)

train_data = train_dataset.datas.numpy()
train_label = train_dataset.targets.numpy()
val_data = val_dataset.datas.numpy()
val_label = val_dataset.targets.numpy()


rf = RandomForestClassifier(
    n_estimators=100, max_depth=10, random_state=0, n_jobs=16, class_weight="balanced"
)
rf.fit(train_data, train_label)
y_pred = rf.predict(val_data)
print("accuracy:", accuracy_score(val_label, y_pred))
print("f1:", f1_score(val_label, y_pred))

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
importances = np.around(importances, decimals=4)
print(importances)
print(list(indices))
print()


keep = 9  # we have 9 features have importance>0.01
print(f"last {keep} features")
new_train_data = train_data[:, indices[:keep]]
new_val_data = val_data[:, indices[:keep]]

rf = RandomForestClassifier(
    n_estimators=10,
    max_depth=110,
    random_state=0,
    n_jobs=16,
    class_weight="balanced",
)
rf.fit(new_train_data, train_label)
y_pred = rf.predict(new_val_data)
print("accuracy:", accuracy_score(val_label, y_pred))
print("f1:", f1_score(val_label, y_pred))
print()
