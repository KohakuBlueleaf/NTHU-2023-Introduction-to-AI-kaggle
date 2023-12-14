import csv

import torch

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from data.dataset import Dataset


for i in range(5):
    train_dataset = Dataset(f"cleaned-train-{i}")
    val_dataset = Dataset(f"cleaned-val-{i}")
    test_dataset = Dataset(f"test-{i}")

    train_data = train_dataset.datas.numpy()
    train_label = train_dataset.targets.numpy()
    val_data = val_dataset.datas.numpy()
    val_label = val_dataset.targets.numpy()
    test_data = test_dataset.datas.numpy()
    test_label = test_dataset.targets.numpy()

    rf = RandomForestClassifier(
        n_estimators=110,
        max_depth=10,
        random_state=0,
        n_jobs=8,
        class_weight="balanced",
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
    new_test_data = test_data[:, indices[:keep]]

    rf = RandomForestClassifier(
        n_estimators=110,
        max_depth=10,
        random_state=0,
        n_jobs=8,
        class_weight="balanced",
    )
    rf.fit(new_train_data, train_label)
    y_pred = rf.predict(new_val_data)
    print("accuracy:", accuracy_score(val_label, y_pred))
    print("f1:", f1_score(val_label, y_pred))
    print()

    result = rf.predict(new_test_data)

    with open(f"out/submission{i}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label"])
        for i in range(len(result)):
            writer.writerow([i, result[i]])
