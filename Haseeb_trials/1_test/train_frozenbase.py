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


#python train_frozenbase.py --data-dir data/pizza_steak_sushi


# python train_frozenbase.py --dataset cifar10 --head linear
# python train_frozenbase.py --dataset cifar10 --head quadratic

# python train_frozenbase.py --dataset flowers102 --head linear
# python train_frozenbase.py --dataset flowers102 --head quadratic

# python train_frozenbase.py --dataset pets --head linear
# python train_frozenbase.py --dataset pets --head quadratic

# python train_frozenbase.py --dataset food101 --head linear
# python train_frozenbase.py --dataset food101 --head quadratic

# python train_frozenbase.py --dataset cifar10 --head linear
# python train_frozenbase.py --dataset cifar10 --head lora
# python train_frozenbase.py --dataset cifar10 --head quadratic

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class QuadraticHead(nn.Module):
#     def __init__(self, in_features: int, num_classes: int):
#         super().__init__()
#         self.linear = nn.Linear(in_features, num_classes)
#         self.quad = nn.Parameter(torch.zeros(num_classes, in_features, in_features))

#     def forward(self, x):
#         # x: [batch, d]
#         linear_out = self.linear(x)  # [batch, C]

#         # quadratic_out[c] = x^T Q_c x
#         quadratic_out = torch.einsum('bi,cij,bj->bc', x, self.quad, x)

#         return linear_out + quadratic_out


class QuadraticAdapterHead(nn.Module):
    """
    QuadraNet V2-style classifier head:
        y = base_linear(x) + quad_adapter(x)

    quad_adapter_c(x) = sum_r (a[c,r]·x) * (b[c,r]·x)

    This is a low-rank quadratic form per class.
    """
    def __init__(self, base_linear: nn.Linear, rank: int = 4, freeze_base: bool = True):
        super().__init__()
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.rank = rank

        # Reuse pretrained / existing linear classifier term
        self.base = nn.Linear(
            self.in_features,
            self.out_features,
            bias=base_linear.bias is not None
        )
        self.base.load_state_dict(base_linear.state_dict())

        if freeze_base:
            for p in self.base.parameters():
                p.requires_grad = False

        # Low-rank quadratic adapter:
        # Wa, Wb: [C, r, d]
        self.wa = nn.Parameter(torch.zeros(self.out_features, rank, self.in_features))
        self.wb = nn.Parameter(torch.empty(self.out_features, rank, self.in_features))

        # Zero-output init:
        # quad(x) = 0 at start because wa = 0
        nn.init.normal_(self.wb, mean=0.0, std=0.02)

    def forward(self, x):
        # base linear term
        linear_out = self.base(x)  # [B, C]

        # ax, bx: [B, C, r]
        ax = torch.einsum("bd,crd->bcr", x, self.wa)
        bx = torch.einsum("bd,crd->bcr", x, self.wb)

        # sum over rank dimension -> [B, C]
        quad_out = (ax * bx).sum(dim=-1)

        return linear_out + quad_out
    
class LoRAAdapterHead(nn.Module):
    """
    LoRA-style classifier head:
        y = base_linear(x) + scale * B(Ax)

    Equivalent to:
        y = (W0 + ΔW) x + b
        ΔW = B @ A

    Added trainable params:
        A: [r, d]
        B: [C, r]
    """
    def __init__(self, base_linear: nn.Linear, rank: int = 4, alpha: float = 4.0, freeze_base: bool = False):
        super().__init__()
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.rank = rank
        self.scaling = alpha / rank

        self.base = nn.Linear(
            self.in_features,
            self.out_features,
            bias=base_linear.bias is not None
        )
        self.base.load_state_dict(base_linear.state_dict())

        if freeze_base:
            for p in self.base.parameters():
                p.requires_grad = False

        # LoRA factors: ΔW = B @ A
        self.A = nn.Parameter(torch.empty(rank, self.in_features))
        self.B = nn.Parameter(torch.zeros(self.out_features, rank))

        # Standard LoRA init: A random, B zero -> zero initial update
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def forward(self, x):
        base_out = self.base(x)                         # [B, C]
        lora_out = (x @ self.A.t()) @ self.B.t()       # [B, C]
        return base_out + self.scaling * lora_out
    
# def create_dataloadersold(data_dir: Path, batch_size: int, num_workers: int):
#     train_dir = data_dir / "train"
#     test_dir = data_dir / "test"

#     #weights = EfficientNet_B0_Weights.DEFAULT
#     weights = ResNet18_Weights.DEFAULT
#     auto_transforms = weights.transforms()

#     train_data = datasets.ImageFolder(root=train_dir, transform=auto_transforms)
#     test_data = datasets.ImageFolder(root=test_dir, transform=auto_transforms)

#     train_loader = DataLoader(
#         train_data,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=num_workers,
#         pin_memory=torch.cuda.is_available(),
#     )

#     test_loader = DataLoader(
#         test_data,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=num_workers,
#         pin_memory=torch.cuda.is_available(),
#     )

#     return train_loader, test_loader, train_data.classes

def create_model(num_classes: int, device: torch.device, head_type: str) -> nn.Module:
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)

    for param in model.parameters():
        param.requires_grad = False

    in_features = model.fc.in_features

    if head_type == "linear":
        model.fc = nn.Linear(in_features, num_classes)

    elif head_type == "mlp":
        model.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    elif head_type == "quadratic":
        base_linear = nn.Linear(in_features, num_classes)
        model.fc = QuadraticAdapterHead(base_linear, rank=4, freeze_base=True)

    elif head_type == "lora":
        base_linear = nn.Linear(in_features, num_classes)
        model.fc = LoRAAdapterHead(base_linear, rank=4, alpha=4.0, freeze_base=False)

    else:
        raise ValueError(f"Unknown head_type: {head_type}")

    return model.to(device)


def get_dataset_pair(dataset_name: str, root: Path, transform):
    dataset_name = dataset_name.lower()

    if dataset_name == "cifar10":
        train_data = datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
        test_data = datasets.CIFAR10(root=root, train=False, download=True, transform=transform)
        class_names = train_data.classes

    elif dataset_name == "cifar100":
        train_data = datasets.CIFAR100(root=root, train=True, download=True, transform=transform)
        test_data = datasets.CIFAR100(root=root, train=False, download=True, transform=transform)
        class_names = train_data.classes

    elif dataset_name == "flowers102":
        train_data = datasets.Flowers102(root=root, split="train", download=True, transform=transform)
        test_data = datasets.Flowers102(root=root, split="test", download=True, transform=transform)
        class_names = [str(i) for i in range(102)]

    elif dataset_name == "pets":
        train_data = datasets.OxfordIIITPet(root=root, split="trainval", download=True, transform=transform)
        test_data = datasets.OxfordIIITPet(root=root, split="test", download=True, transform=transform)
        class_names = train_data.classes

    elif dataset_name == "food101":
        train_data = datasets.Food101(root=root, split="train", download=True, transform=transform)
        test_data = datasets.Food101(root=root, split="test", download=True, transform=transform)
        class_names = train_data.classes

    else:
        raise ValueError(
            "dataset_name must be one of: "
            "cifar10, cifar100, flowers102, pets, food101"
        )

    return train_data, test_data, class_names


def create_dataloaders(dataset_name: str, root: Path, batch_size: int, num_workers: int):
    weights = ResNet18_Weights.DEFAULT
    auto_transforms = weights.transforms()

    train_data, test_data, class_names = get_dataset_pair(
        dataset_name=dataset_name,
        root=root,
        transform=auto_transforms,
    )

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

    return train_loader, test_loader, class_names

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


def save_checkpoint(model, classes, output_dir: Path, filename: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "class_names": classes,
    }
    torch.save(checkpoint, output_dir / filename)


def report_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["cifar10", "cifar100", "flowers102", "pets", "food101"])
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--head", type=str, default="linear",
                    choices=["linear", "mlp", "quadratic", "lora"])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()

    output_dir = Path(args.output_dir) / args.dataset / args.head
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader, class_names = create_dataloaders(
        dataset_name=args.dataset,
        root=Path(args.data_root),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = create_model(
        num_classes=len(class_names),
        device=device,
        head_type=args.head,
    )
    
    print(f"Trainable params: {report_params(model):,}")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
    )

    history = {
        "dataset": args.dataset,
        "head": args.head,
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }

    start_time = time.time()

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, loss_fn, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        print(
            f"[{args.dataset}][{args.head}] "
            f"Epoch {epoch+1}/{args.epochs} | "
            f"train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | test_acc: {test_acc:.4f}"
        )

    elapsed = time.time() - start_time
    print(f"Training time: {elapsed:.2f} seconds")

    save_checkpoint(model, class_names, output_dir, "model.pt")

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()