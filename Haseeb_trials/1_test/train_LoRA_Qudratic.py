import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.models import resnet18, ResNet18_Weights


# ------------------------------------------------------------
# Repro / device
# ------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------
# Adapter heads
# ------------------------------------------------------------

class QuadraticAdapterHead(nn.Module):
    """
    Frozen base linear term + trainable low-rank quadratic adapter.

    y = base(x) + quad(x)

    quad_c(x) = sum_r (a[c,r]·x) * (b[c,r]·x)

    Trainable params:
        wa: [C, r, d]
        wb: [C, r, d]

    Zero-output init:
        wa = 0, wb ~ N(0, 0.02), so quad(x) = 0 initially.
    """
    def __init__(self, base_linear: nn.Linear, rank: int = 4, freeze_base: bool = True):
        super().__init__()
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.rank = rank

        self.base = nn.Linear(
            self.in_features,
            self.out_features,
            bias=base_linear.bias is not None,
        )
        self.base.load_state_dict(base_linear.state_dict())

        if freeze_base:
            for p in self.base.parameters():
                p.requires_grad = False

        self.wa = nn.Parameter(torch.zeros(self.out_features, rank, self.in_features))
        self.wb = nn.Parameter(torch.empty(self.out_features, rank, self.in_features))
        nn.init.normal_(self.wb, mean=0.0, std=0.02)

    def forward(self, x):
        linear_out = self.base(x)  # [B, C]
        ax = torch.einsum("bd,crd->bcr", x, self.wa)  # [B, C, r]
        bx = torch.einsum("bd,crd->bcr", x, self.wb)  # [B, C, r]
        quad_out = (ax * bx).sum(dim=-1)              # [B, C]
        return linear_out + quad_out


class LoRAAdapterHead(nn.Module):
    """
    Frozen base linear term + trainable LoRA update.

    y = base(x) + scaling * B(Ax)

    DeltaW = B @ A
      A: [r, d]
      B: [C, r]

    Zero-output init:
      A random, B = 0, so the LoRA branch starts at 0.
    """
    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int = 4,
        alpha: float = None,
        freeze_base: bool = True,
    ):
        super().__init__()
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.rank = rank
        self.alpha = float(rank) if alpha is None else float(alpha)
        self.scaling = self.alpha / self.rank

        self.base = nn.Linear(
            self.in_features,
            self.out_features,
            bias=base_linear.bias is not None,
        )
        self.base.load_state_dict(base_linear.state_dict())

        if freeze_base:
            for p in self.base.parameters():
                p.requires_grad = False

        self.A = nn.Parameter(torch.empty(rank, self.in_features))
        self.B = nn.Parameter(torch.zeros(self.out_features, rank))

        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def forward(self, x):
        base_out = self.base(x)                    # [B, C]
        lora_out = (x @ self.A.t()) @ self.B.t()  # [B, C]
        return base_out + self.scaling * lora_out


# ------------------------------------------------------------
# Dataset helpers
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Checkpoint / parameter helpers
# ------------------------------------------------------------

def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def report_params(model: nn.Module) -> None:
    trainable = count_trainable_params(model)
    total = count_total_params(model)
    pct = 100.0 * trainable / total
    print(f"Trainable params: {trainable:,} / {total:,} ({pct:.2f}%)")


def print_trainable_parameter_names(model: nn.Module) -> None:
    print("Trainable parameter tensors:")
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(f"  {name:<30} {p.numel():>10,}")


def matched_lora_rank(in_features: int, num_classes: int, quad_rank: int) -> int:
    """
    Match LoRA head trainable params roughly to quadratic head trainable params.

    Quadratic params = 2 * C * rq * d
    LoRA params      = rl * (d + C)

    rl ~= (2 * C * rq * d) / (d + C)
    """
    rank = round((2 * num_classes * quad_rank * in_features) / (in_features + num_classes))
    return max(1, rank)


def extract_linear_head_state_dict(model: nn.Module, stage: str):
    if stage == "linear_base":
        return model.fc.state_dict()
    elif stage in ("quadratic_adapter", "lora_adapter"):
        return model.fc.base.state_dict()
    else:
        raise ValueError(f"Unknown stage: {stage}")


def extract_in_features(model: nn.Module, stage: str) -> int:
    if stage == "linear_base":
        return model.fc.in_features
    elif stage in ("quadratic_adapter", "lora_adapter"):
        return model.fc.base.in_features
    else:
        raise ValueError(f"Unknown stage: {stage}")


def load_linear_head_from_checkpoint(
    checkpoint_path: Path,
    in_features: int,
    num_classes: int,
    device: torch.device,
) -> nn.Linear:
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "linear_head_state_dict" not in checkpoint:
        raise ValueError(
            f"{checkpoint_path} does not contain 'linear_head_state_dict'. "
            "Make sure it is a checkpoint saved from stage='linear_base'."
        )

    ckpt_num_classes = checkpoint.get("num_classes")
    ckpt_in_features = checkpoint.get("in_features")

    if ckpt_num_classes is not None and ckpt_num_classes != num_classes:
        raise ValueError(
            f"Checkpoint num_classes={ckpt_num_classes}, "
            f"but current dataset has num_classes={num_classes}."
        )

    if ckpt_in_features is not None and ckpt_in_features != in_features:
        raise ValueError(
            f"Checkpoint in_features={ckpt_in_features}, "
            f"but current model expects in_features={in_features}."
        )

    base_linear = nn.Linear(in_features, num_classes)
    base_linear.load_state_dict(checkpoint["linear_head_state_dict"])
    return base_linear


def save_checkpoint(
    model: nn.Module,
    classes,
    output_dir: Path,
    filename: str,
    stage: str,
    args_dict: dict,
    resolved_lora_rank: int = None,
    resolved_lora_alpha: float = None,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "stage": stage,
        "model_state_dict": model.state_dict(),
        "linear_head_state_dict": extract_linear_head_state_dict(model, stage),
        "class_names": classes,
        "num_classes": len(classes),
        "in_features": extract_in_features(model, stage),
        "trainable_params": count_trainable_params(model),
        "args": args_dict,
        "resolved_lora_rank": resolved_lora_rank,
        "resolved_lora_alpha": resolved_lora_alpha,
    }

    torch.save(checkpoint, output_dir / filename)


# ------------------------------------------------------------
# Model creation
# ------------------------------------------------------------

def create_model(
    num_classes: int,
    device: torch.device,
    stage: str,
    base_checkpoint: Path = None,
    quad_rank: int = 4,
    lora_rank: int = None,
    lora_alpha: float = None,
):
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)

    # Freeze the ImageNet-pretrained backbone
    for param in model.parameters():
        param.requires_grad = False

    in_features = model.fc.in_features
    resolved_lora_rank = lora_rank
    resolved_lora_alpha = lora_alpha

    if stage == "linear_base":
        model.fc = nn.Linear(in_features, num_classes)

    elif stage == "quadratic_adapter":
        if base_checkpoint is None:
            raise ValueError("--base-checkpoint is required for stage='quadratic_adapter'")

        base_linear = load_linear_head_from_checkpoint(
            checkpoint_path=base_checkpoint,
            in_features=in_features,
            num_classes=num_classes,
            device=device,
        )
        model.fc = QuadraticAdapterHead(
            base_linear=base_linear,
            rank=quad_rank,
            freeze_base=True,
        )

    elif stage == "lora_adapter":
        if base_checkpoint is None:
            raise ValueError("--base-checkpoint is required for stage='lora_adapter'")

        base_linear = load_linear_head_from_checkpoint(
            checkpoint_path=base_checkpoint,
            in_features=in_features,
            num_classes=num_classes,
            device=device,
        )

        if resolved_lora_rank is None:
            resolved_lora_rank = matched_lora_rank(
                in_features=in_features,
                num_classes=num_classes,
                quad_rank=quad_rank,
            )
            print(
                f"[info] --lora-rank not provided; using matched rank {resolved_lora_rank} "
                f"to roughly match quadratic rank {quad_rank}."
            )

        if resolved_lora_alpha is None:
            resolved_lora_alpha = float(resolved_lora_rank)

        model.fc = LoRAAdapterHead(
            base_linear=base_linear,
            rank=resolved_lora_rank,
            alpha=resolved_lora_alpha,
            freeze_base=True,
        )

    else:
        raise ValueError(f"Unknown stage: {stage}")

    return model.to(device), resolved_lora_rank, resolved_lora_alpha


# ------------------------------------------------------------
# Train / eval
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["cifar10", "cifar100", "flowers102", "pets", "food101"],
    )
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=["linear_base", "quadratic_adapter", "lora_adapter"],
    )

    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--base-checkpoint", type=str, default=None)

    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--quad-rank", type=int, default=4)
    parser.add_argument("--lora-rank", type=int, default=None)
    parser.add_argument("--lora-alpha", type=float, default=None)

    parser.add_argument(
        "--print-trainable-names",
        action="store_true",
        help="Print names and sizes of all trainable parameter tensors.",
    )

    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()

    train_loader, test_loader, class_names = create_dataloaders(
        dataset_name=args.dataset,
        root=Path(args.data_root),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    base_checkpoint = None if args.base_checkpoint is None else Path(args.base_checkpoint)

    model, resolved_lora_rank, resolved_lora_alpha = create_model(
        num_classes=len(class_names),
        device=device,
        stage=args.stage,
        base_checkpoint=base_checkpoint,
        quad_rank=args.quad_rank,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
    )

    # Output dir
    output_dir = Path(args.output_dir) / args.dataset / args.stage
    if args.stage == "quadratic_adapter":
        output_dir = output_dir / f"rank_{args.quad_rank}"
    elif args.stage == "lora_adapter":
        output_dir = output_dir / f"rank_{resolved_lora_rank}_alpha_{resolved_lora_alpha:g}"
    output_dir.mkdir(parents=True, exist_ok=True)

    report_params(model)
    if args.print_trainable_names:
        print_trainable_parameter_names(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
    )

    history = {
        "dataset": args.dataset,
        "stage": args.stage,
        "base_checkpoint": str(base_checkpoint) if base_checkpoint is not None else None,
        "quad_rank": args.quad_rank,
        "resolved_lora_rank": resolved_lora_rank,
        "resolved_lora_alpha": resolved_lora_alpha,
        "trainable_params": count_trainable_params(model),
        "total_params": count_total_params(model),
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
            f"[{args.dataset}][{args.stage}] "
            f"Epoch {epoch+1}/{args.epochs} | "
            f"train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | test_acc: {test_acc:.4f}"
        )

    elapsed = time.time() - start_time
    history["training_time_sec"] = elapsed
    print(f"Training time: {elapsed:.2f} seconds")

    args_dict = dict(vars(args))
    if args_dict["base_checkpoint"] is not None:
        args_dict["base_checkpoint"] = str(args_dict["base_checkpoint"])

    save_checkpoint(
        model=model,
        classes=class_names,
        output_dir=output_dir,
        filename="model.pt",
        stage=args.stage,
        args_dict=args_dict,
        resolved_lora_rank=resolved_lora_rank,
        resolved_lora_alpha=resolved_lora_alpha,
    )

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
    
    
# # 1) Train the frozen-backbone linear base
# python train_LoRA_Qudratic.py \
#   --dataset cifar10 \
#   --stage linear_base \
#   --epochs 10 \
#   --lr 1e-3

# # 2) Train only the quadratic adapter on top of that same frozen linear base
# python train_LoRA_Qudratic.py \
#   --dataset cifar10 \
#   --stage quadratic_adapter \
#   --base-checkpoint outputs/cifar10/linear_base/model.pt \
#   --quad-rank 4 \
#   --epochs 10 \
#   --lr 1e-3

# # 3) Train only the LoRA adapter on top of that same frozen linear base
# # If you omit --lora-rank, the script auto-picks one that roughly matches
# # the quadratic head's trainable-parameter budget.
# python train_LoRA_Qudratic.py \
#   --dataset cifar10 \
#   --stage lora_adapter \
#   --base-checkpoint outputs/cifar10/linear_base/model.pt \
#   --quad-rank 4 \
#   --epochs 10 \
#   --lr 1e-3