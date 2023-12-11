import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

from pytorch_lightning import seed_everything
from prodigyopt import Prodigy

from modules.model import FEClassifier
from data.dataset import Dataset
from metadatas import indices

seed_everything(0)


EPOCH = 10
BATCH_SIZE = 1024
KEEP_FEATURES = 24


def main():
    mask = [i for i in range(35) if i in indices[:KEEP_FEATURES]]
    train_dataset = Dataset("cleaned-train", mask)
    train_dataset.balancing()
    valid_dataset = Dataset("cleaned-val", mask)

    mu, sigma = torch.load("./data/standardize_factor.pth")
    train_dataset.standardize(mu, sigma)
    valid_dataset.standardize(mu, sigma)

    # mu, x_min, x_max = torch.load('./normalize_factor.pth')
    # train_dataset.normalize(mu, x_min.values, x_max.values)
    # valid_dataset.normalize(mu, x_min.values, x_max.values)

    loader = data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )
    val_loader = data.DataLoader(valid_dataset, batch_size=2000)
    total_step = len(loader) * EPOCH

    net = FEClassifier(KEEP_FEATURES, 2, 128, weighted_residual=True).cuda()
    criterion = nn.CrossEntropyLoss(torch.tensor([1, 1.2]), label_smoothing=0.1).cuda()
    optimizer = Prodigy(net.parameters(), lr=1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_step,
        eta_min=1e-2,
    )

    ema_loss = 0.0
    step = 0
    max_acc = 0

    for epoch in range(EPOCH):
        total = 0
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0
        net.train()
        for i, (inputs, labels) in enumerate(loader):
            # Training step
            with torch.autocast("cuda", torch.bfloat16):
                inputs = inputs.cuda()
                outputs = net(inputs)
            predicted = torch.argmax(outputs.data, 1)

            loss = criterion(outputs.float(), labels.cuda())
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # Logging
            total += labels.size(0)
            true_positive += ((predicted == 1) & (labels.cuda() == 1)).sum().item()
            true_negative += ((predicted == 0) & (labels.cuda() == 0)).sum().item()
            false_positive += ((predicted == 1) & (labels.cuda() == 0)).sum().item()
            false_negative += ((predicted == 0) & (labels.cuda() == 1)).sum().item()

            correct = true_positive + true_negative
            acc = correct / total
            precision = true_positive / max(true_positive + false_positive, 1)
            recall = true_positive / max(true_positive + false_negative, 1)
            f1_score = 2 * precision * recall / max(precision + recall, 1e-8)

            ema_decay = min(0.999, step / 1000)
            ema_loss = ema_loss * ema_decay + loss.item() * (1 - ema_decay)
            step += 1

            if i % 10 == 0:
                print(
                    f"[{epoch + 1}, {i:5d}]"
                    f'| d*lr: {optimizer.param_groups[0]["d"] * optimizer.param_groups[0]["lr"]:.5f}'
                    f"| loss: {ema_loss:.3f}"
                    f"| acc: {correct / total * 100:.3f}%",
                    f"| f1: {f1_score:.5f}",
                    end="\r",
                )
        print(
            f"[{epoch + 1}, {i:5d}] loss: {ema_loss:.3f} | acc: {correct / total * 100:.3f}"
        )

        net.eval()
        # validation
        total = 0
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(val_loader):
                with torch.autocast("cuda", torch.bfloat16):
                    outputs = net(inputs.cuda())
                predicted = torch.argmax(outputs.data, 1)
                total += labels.size(0)
                true_positive += ((predicted == 1) & (labels.cuda() == 1)).sum().item()
                true_negative += ((predicted == 0) & (labels.cuda() == 0)).sum().item()
                false_positive += ((predicted == 1) & (labels.cuda() == 0)).sum().item()
                false_negative += ((predicted == 0) & (labels.cuda() == 1)).sum().item()
        correct = true_positive + true_negative
        acc = correct / total
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        f1_score = 2 * precision * recall / (precision + recall)
        print(f"Validation Accuracy: {acc*100:.3f} %")
        print(f"Validation Precision: {precision*100:.3f} %")
        print(f"Validation Recall: {recall*100:.3f} %")
        print(f"Validation F1 Score: {f1_score:.5f}")
        if f1_score > max_acc:
            max_acc = f1_score
            print(f"Best F1 achieved, save intermediate model to model.pth")
            torch.save(net.state_dict(), "best-model.pth")
        print()

    torch.save(net.state_dict(), "final-model.pth")


if __name__ == "__main__":
    main()
