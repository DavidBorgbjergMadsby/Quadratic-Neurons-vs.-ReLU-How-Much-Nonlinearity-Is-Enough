import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
#from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models import resnet18, ResNet18_Weights


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_dataloaders(data_dir: Path, batch_size: int, num_workers: int):
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"

    #weights = EfficientNet_B0_Weights.DEFAULT
    weights = ResNet18_Weights.DEFAULT
    auto_transforms = weights.transforms()

    train_data = datasets.ImageFolder(root=train_dir, transform=auto_transforms)
    test_data = datasets.ImageFolder(root=test_dir, transform=auto_transforms)

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, test_loader, train_data.classes


class QuadraticHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)
        self.quad = nn.Parameter(torch.zeros(num_classes, in_features, in_features))

    def forward(self, x):
        # x: [batch, d]
        linear_out = self.linear(x)  # [batch, C]

        # quadratic_out[c] = x^T Q_c x
        quadratic_out = torch.einsum('bi,cij,bj->bc', x, self.quad, x)

        return linear_out + quadratic_out

# def create_model(num_classes: int, device: torch.device) -> nn.Module:
#     weights = EfficientNet_B0_Weights.DEFAULT
#     model = efficientnet_b0(weights=weights)
    
#     for param in model.features.parameters():
#         param.requires_grad = False

#     model.classifier = nn.Sequential(
#         nn.Dropout(p=0.2, inplace=True),
#         nn.Linear(in_features=1280, out_features=num_classes),
#     )

#     return model.to(device)

def create_model(num_classes: int, device: torch.device) -> nn.Module:
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)

    for param in model.parameters():
        param.requires_grad = False

    #model.fc = nn.Linear(model.fc.in_features, num_classes)            #original 
    
    # model.fc = nn.Sequential(                     #relu normal
    #     nn.Linear(model.fc.in_features, 256),
    #     nn.ReLU(),
    #     nn.Dropout(0.3),
    #     nn.Linear(256, num_classes)
    # )
    
    in_features = model.fc.in_features      #quandratic 
    model.fc = QuadraticHead(in_features, num_classes)

    return model.to(device)

def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * X.size(0)
        preds = y_pred.argmax(dim=1)
        train_correct += (preds == y).sum().item()
        train_total += y.size(0)

    avg_loss = train_loss / train_total
    avg_acc = train_correct / train_total
    return avg_loss, avg_acc


@torch.inference_mode()
def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        y_pred = model(X)
        loss = loss_fn(y_pred, y)

        test_loss += loss.item() * X.size(0)
        preds = y_pred.argmax(dim=1)
        test_correct += (preds == y).sum().item()
        test_total += y.size(0)

    avg_loss = test_loss / test_total
    avg_acc = test_correct / test_total
    return avg_loss, avg_acc


def save_checkpoint(model, classes, output_dir: Path, filename: str = "model.pt"):
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "class_names": classes,
    }
    torch.save(checkpoint, output_dir / filename)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader, class_names = create_dataloaders(
        data_dir=data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = create_model(num_classes=len(class_names), device=device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }

    start_time = time.time()

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device
        )
        test_loss, test_acc = evaluate(
            model, test_loader, loss_fn, device
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | test_acc: {test_acc:.4f}"
        )

    elapsed = time.time() - start_time
    print(f"Training time: {elapsed:.2f} seconds")

    save_checkpoint(model, class_names, output_dir, "efficientnet_b0_transfer.pt")

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()