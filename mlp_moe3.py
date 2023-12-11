from sklearn.metrics import f1_score
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm, trange
from lamb import Lamb
from prodigyopt import Prodigy

# Load data
train_data = torch.Tensor(np.load("data/train_data.npy"))
train_label = torch.LongTensor(np.load("data/train_label.npy"))
test_data = torch.Tensor(np.load("data/test_data.npy"))
test_label = torch.LongTensor(np.load("data/test_label.npy"))

# indices = [2, 1, 9, 19, 21, 3, 12, 13, 22, 33, 6, 18, 24, 5, 26, 8, 4, 28, 27, 11, 0, 16, 30, 15,
#  31, 32, 34, 29, 14, 17, 23, 25, 7, 20, 10]
# keep_k = 9

# print('Keep', keep_k, 'features at index', indices[:keep_k])

new_train_data = train_data
new_test_data = test_data

# Create data loaders
batch_size = 512

# We cluster the data into 5 clusters.
# For each cluster, we train a model. Finally, we use corresponding model to predict the test data.
kmeans = KMeans(n_clusters=5, random_state=0, n_init="auto").fit(new_train_data)
train_data_cluster = kmeans.predict(new_train_data)
kmeans = KMeans(n_clusters=5, random_state=0, n_init="auto").fit(new_test_data)
test_data_cluster = kmeans.predict(new_test_data)

# We always train on all data
train_loader = DataLoader(
    TensorDataset(new_train_data, train_label), batch_size=batch_size, shuffle=True
)

test_data_list = []
test_label_list = []
for i in range(5):
    test_data_list.append(new_test_data[test_data_cluster == i])
    test_label_list.append(test_label[test_data_cluster == i])


# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        return self.model(x)


# Define Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction="mean"):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = torch.tensor([0.11, 0.89], device="cuda")
        else:
            self.alpha = alpha.cuda()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = nn.functional.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-BCE_loss)
        alpha = self.alpha[targets]
        F_loss = alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == "mean":
            return torch.mean(F_loss)
        elif self.reduction == "sum":
            return torch.sum(F_loss)
        else:
            return F_loss


# Initialize the MLP
input_size = new_train_data.shape[1]
hidden_size = 256  # You can tune this
num_classes = len(torch.unique(train_label))
model = MLP(input_size, hidden_size, num_classes).cuda()

alpha = torch.tensor([0.1, 0.9], device="cuda")
criterion = FocalLoss(alpha=alpha, gamma=3)
# Loss and optimizer
optimizer = Prodigy(model.parameters(), lr=0.1)
# Train the model
num_epochs = 100  # You can tune this
best_test_acc = [0] * 5
best_test_f1 = [0] * 5
for epoch in range(num_epochs):
    train_acc = []
    model.train()
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        # Forward pass
        outputs = model(inputs.cuda())
        loss = criterion(outputs, labels.cuda())
        train_acc.append(
            (torch.argmax(outputs, dim=1) == labels.cuda()).sum().item()
            / labels.size(0)
        )
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    for i in range(5):
        # Test the model
        model.eval()
        with torch.inference_mode():
            test_outputs = model(test_data_list[i].cuda())
            test_acc = (
                torch.argmax(test_outputs, dim=1) == test_label_list[i].cuda()
            ).sum().item() / test_label_list[i].size(0)
            test_f1 = f1_score(
                test_label_list[i].cpu(), torch.argmax(test_outputs, dim=1).cpu()
            )
            if test_acc > best_test_acc[i]:
                best_test_acc[i] = test_acc
                best_test_f1[i] = test_f1
                torch.save(model.state_dict(), f"ckpt/mlp_moe_{i}.pt")

            print(
                f"Epoch {epoch+1}/{num_epochs}, Cluster {i}, Train Accuracy: {np.mean(train_acc):.4f}, Test Accuracy: {test_acc:.4f}, Test F1: {test_f1:.4f}"
            )

    print("Best Test Accuracies:", best_test_acc, "Best Test F1s:", best_test_f1)


models = [MLP(input_size, hidden_size, num_classes).cuda() for _ in range(5)]
# Test the model
test_outputs = []
test_labels = []
for i in range(5):
    model = models[i]
    model.load_state_dict(torch.load(f"ckpt/mlp_moe_{i}.pt"))
    model.eval()
    with torch.inference_mode():
        current_test_outputs = model(test_data_list[i].cuda())
        test_outputs.append(current_test_outputs)
        current_test_labels = test_label_list[i].cuda()
        test_labels.append(current_test_labels)

        accuracy = (
            torch.argmax(current_test_outputs, dim=1) == current_test_labels
        ).sum().item() / current_test_labels.size(0)
        f1 = f1_score(
            current_test_labels.cpu(), torch.argmax(current_test_outputs, dim=1).cpu()
        )
        print(f"Model {i} Test Accuracy: {accuracy:.4f}, Test F1: {f1:.4f}")

test_outputs = torch.cat(test_outputs, dim=0)
test_labels = torch.cat(test_labels, dim=0)
_, predicted = torch.max(test_outputs.data, 1)
accuracy = (predicted == test_labels).sum().item() / test_labels.size(0)
f1 = f1_score(test_labels.cpu().numpy(), predicted.cpu().numpy(), average="macro")

print(f"Final Test Accuracy: {accuracy:.4f}, Final Test F1: {f1:.4f}")
