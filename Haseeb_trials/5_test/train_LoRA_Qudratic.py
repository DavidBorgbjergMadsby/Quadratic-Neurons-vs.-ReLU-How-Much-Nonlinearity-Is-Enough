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
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights


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
    weights = ConvNeXt_Tiny_Weights.DEFAULT
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
    if stage in ("linear_base", "scratch"):
        return model.fc.state_dict()
    elif stage in ("quadratic_adapter", "lora_adapter"):
        return model.fc.base.state_dict()
    else:
        raise ValueError(f"Unknown stage: {stage}")


def extract_in_features(model: nn.Module, stage: str) -> int:
    if stage in ("linear_base", "scratch"):
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
    include_model_state: bool = False,
):
    """
    Save a compact checkpoint by default.

    For linear_base, quad/lora only need classifier_state_dict, num_classes,
    and in_features. The full ConvNeXt model_state_dict is large and is not
    needed as the adapter base checkpoint.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    classifier = get_convnext_classifier(model)

    checkpoint = {
        "stage": stage,
        "classifier_state_dict": classifier.state_dict(),
        "class_names": classes,
        "num_classes": len(classes),
        "in_features": classifier.in_features,
        "trainable_params": count_trainable_params(model),
        "total_params": count_total_params(model),
        "args": args_dict,
    }

    if include_model_state:
        checkpoint["model_state_dict"] = model.state_dict()

    torch.save(checkpoint, output_dir / filename)


def is_depthwise_conv(m: nn.Module) -> bool:
    return (
        isinstance(m, nn.Conv2d)
        and m.groups == m.in_channels
        and m.in_channels == m.out_channels
        and m.kernel_size != (1, 1)
    )


def replace_depthwise_convs(
    module: nn.Module,
    adapter_type: str,
    rank: int,
    alpha: float = None,
    quad_adapter_kernel_size: int = None,
    quad_adapter_dilation: int = None,
) -> int:
    replaced = 0

    for name, child in list(module.named_children()):
        if is_depthwise_conv(child):
            if adapter_type == "quad":
                setattr(
                    module,
                    name,
                    QuadraticDWConvAdapter(
                        child,
                        rank=rank,
                        alpha=alpha,
                        adapter_kernel_size=quad_adapter_kernel_size,
                        adapter_dilation=quad_adapter_dilation,
                    ),
                )
            elif adapter_type == "lora":
                setattr(module, name, LoRADWConvAdapter(child, rank=rank, alpha=alpha))
            else:
                raise ValueError(f"Unknown adapter_type: {adapter_type}")

            replaced += 1
        else:
            replaced += replace_depthwise_convs(
                child,
                adapter_type,
                rank,
                alpha,
                quad_adapter_kernel_size,
                quad_adapter_dilation,
            )

    return replaced


class LoRADWConvAdapter(nn.Module):
    """
    LoRA-style linear adapter around a frozen depthwise conv.

    y = frozen_dwconv(x) + scale * up(down(x))
    """
    def __init__(self, base_conv: nn.Conv2d, rank: int = 4, alpha: float = None):
        super().__init__()
        assert base_conv.groups == base_conv.in_channels == base_conv.out_channels

        self.base = base_conv
        for p in self.base.parameters():
            p.requires_grad = False

        C = base_conv.in_channels
        self.rank = rank
        self.alpha = float(rank) if alpha is None else float(alpha)
        self.scaling = self.alpha / self.rank

        self.down = nn.Conv2d(
            C,
            C * rank,
            kernel_size=base_conv.kernel_size,
            stride=base_conv.stride,
            padding=base_conv.padding,
            dilation=base_conv.dilation,
            groups=C,
            bias=False,
        )

        self.up = nn.Conv2d(
            C * rank,
            C,
            kernel_size=1,
            groups=C,
            bias=False,
        )

        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.base(x) + self.scaling * self.up(self.down(x))


def _pair(value):
    if isinstance(value, tuple):
        return value
    return (value, value)


def _same_padding_for_odd_kernel(kernel_size: int, dilation: int):
    """
    Padding that preserves H/W for stride=1 with odd kernels.
    For ConvNeXt depthwise 7x7 layers, kernel=3 and dilation=3 gives
    effective kernel size 7 and padding 3.
    """
    return dilation * (kernel_size - 1) // 2


class QuadraticDWConvAdapter(nn.Module):
    """
    QuadraNet-style quadratic adapter around a frozen depthwise conv.

    y = frozen_dwconv(x) + scale * sum_r (Wa_r x) * (Wb_r x)

    Wa is zero-initialized, so the adapter starts as exactly the frozen base conv.

    Optional atrous adapter mode:
    For ConvNeXt depthwise 7x7 layers, Wa/Wb can use a smaller 3x3 kernel
    with dilation=3. This keeps a 7x7 effective receptive field while reducing
    adapter parameters and convolution work.
    """
    def __init__(
        self,
        base_conv: nn.Conv2d,
        rank: int = 4,
        alpha: float = None,
        adapter_kernel_size: int = None,
        adapter_dilation: int = None,
    ):
        super().__init__()
        assert base_conv.groups == base_conv.in_channels == base_conv.out_channels

        self.base = base_conv
        for p in self.base.parameters():
            p.requires_grad = False

        C = base_conv.in_channels
        self.C = C
        self.rank = rank
        self.alpha = 1.0 if alpha is None else float(alpha)
        self.scaling = self.alpha / self.rank

        use_atrous_adapter = (
            adapter_kernel_size is not None
            and adapter_dilation is not None
            and _pair(base_conv.kernel_size) == (7, 7)
        )

        if use_atrous_adapter:
            if adapter_kernel_size % 2 != 1:
                raise ValueError("--quad-adapter-kernel-size must be odd for same-padding atrous adapters")

            adapter_kernel = (adapter_kernel_size, adapter_kernel_size)
            adapter_dil = (adapter_dilation, adapter_dilation)
            adapter_pad_value = _same_padding_for_odd_kernel(adapter_kernel_size, adapter_dilation)
            adapter_padding = (adapter_pad_value, adapter_pad_value)
        else:
            adapter_kernel = base_conv.kernel_size
            adapter_dil = base_conv.dilation
            adapter_padding = base_conv.padding

        self.adapter_kernel_size = adapter_kernel
        self.adapter_dilation = adapter_dil
        self.adapter_padding = adapter_padding
        self.using_atrous_adapter = use_atrous_adapter

        self.wa = nn.Conv2d(
            C,
            C * rank,
            kernel_size=adapter_kernel,
            stride=base_conv.stride,
            padding=adapter_padding,
            dilation=adapter_dil,
            groups=C,
            bias=False,
        )

        self.wb = nn.Conv2d(
            C,
            C * rank,
            kernel_size=adapter_kernel,
            stride=base_conv.stride,
            padding=adapter_padding,
            dilation=adapter_dil,
            groups=C,
            bias=False,
        )

        nn.init.zeros_(self.wa.weight)
        nn.init.normal_(self.wb.weight, mean=0.0, std=0.02)

    def forward(self, x):
        base_out = self.base(x)

        a = self.wa(x)
        b = self.wb(x)

        B, CR, H, W = a.shape
        a = a.view(B, self.C, self.rank, H, W)
        b = b.view(B, self.C, self.rank, H, W)

        quad = (a * b).sum(dim=2)
        return base_out + self.scaling * quad



def get_convnext_classifier(model: nn.Module) -> nn.Linear:
    return model.classifier[2]


def set_convnext_classifier(model: nn.Module, num_classes: int) -> None:
    old_head = get_convnext_classifier(model)
    in_features = old_head.in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)


def load_classifier_from_checkpoint(checkpoint_path: Path, model: nn.Module, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "classifier_state_dict" not in checkpoint:
        raise ValueError(
            f"{checkpoint_path} does not contain classifier_state_dict. "
            "Run stage='linear_base' first."
        )

    get_convnext_classifier(model).load_state_dict(checkpoint["classifier_state_dict"])


def create_model(
    num_classes: int,
    device: torch.device,
    stage: str,
    base_checkpoint: Path = None,
    adapter_rank: int = 4,
    adapter_alpha: float = None,
    quad_adapter_kernel_size: int = None,
    quad_adapter_dilation: int = None,
    adapter_scope: str = "last_stage",
    train_head_with_adapter: bool = False,
):
    if stage == "scratch":
        weights = None
    else:
        weights = ConvNeXt_Tiny_Weights.DEFAULT

    model = convnext_tiny(weights=weights)

    # Replace ImageNet classifier with dataset classifier.
    set_convnext_classifier(model, num_classes)

    # Freeze everything by default.
    for p in model.parameters():
        p.requires_grad = False

    if stage == "linear_base":
        # Frozen backbone, train only classifier.
        for p in get_convnext_classifier(model).parameters():
            p.requires_grad = True

    elif stage == "full_finetune":
        # ImageNet-pretrained backbone, train the whole model.
        for p in model.parameters():
            p.requires_grad = True

    elif stage == "scratch":
        # Random initialization, train the whole model.
        for p in model.parameters():
            p.requires_grad = True

    elif stage in ("lora_dw", "quad_dw"):
        if base_checkpoint is None:
            raise ValueError(f"--base-checkpoint is required for stage='{stage}'")

        load_classifier_from_checkpoint(base_checkpoint, model, device)

        if train_head_with_adapter:
            for p in get_convnext_classifier(model).parameters():
                p.requires_grad = True
        else:
            for p in get_convnext_classifier(model).parameters():
                p.requires_grad = False

        if adapter_scope == "last_stage":
            target_module = model.features[-1]
        elif adapter_scope == "all":
            target_module = model.features
        else:
            raise ValueError("--adapter-scope must be 'last_stage' or 'all'")

        if stage == "quad_dw" and ((quad_adapter_kernel_size is None) != (quad_adapter_dilation is None)):
            raise ValueError(
                "Provide both --quad-adapter-kernel-size and --quad-adapter-dilation, or neither."
            )

        adapter_type = "quad" if stage == "quad_dw" else "lora"
        n = replace_depthwise_convs(
            target_module,
            adapter_type,
            adapter_rank,
            adapter_alpha,
            quad_adapter_kernel_size if stage == "quad_dw" else None,
            quad_adapter_dilation if stage == "quad_dw" else None,
        )
        print(
            f"[info] inserted {n} {adapter_type} depthwise adapters "
            f"(rank={adapter_rank}, alpha={adapter_alpha}, "
            f"quad_kernel={quad_adapter_kernel_size}, quad_dilation={quad_adapter_dilation})"
        )

    else:
        raise ValueError(f"Unknown stage: {stage}")

    return model.to(device, memory_format=torch.channels_last)

# ------------------------------------------------------------
# Train / eval
# ------------------------------------------------------------

def train_one_epoch(model, dataloader, loss_fn, optimizer, device, grad_clip: float):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for X, y in dataloader:
        X = X.to(device, memory_format=torch.channels_last)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, model.parameters()),
            max_norm=grad_clip,
        )
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
        X = X.to(device, memory_format=torch.channels_last)
        y = y.to(device)

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
        choices=["linear_base", "full_finetune", "scratch", "lora_dw", "quad_dw"],
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

    parser.add_argument("--adapter-rank", type=int, default=4)
    parser.add_argument("--adapter-alpha", type=float, default=None)
    parser.add_argument(
        "--quad-adapter-kernel-size",
        "--quad-adapter-kernel",
        dest="quad_adapter_kernel_size",
        type=int,
        default=None,
        help="Optional quadratic Wa/Wb kernel size. Use 3 with --quad-adapter-dilation 3 for ConvNeXt 7x7 layers.",
    )
    parser.add_argument(
        "--quad-adapter-dilation",
        type=int,
        default=None,
        help="Optional quadratic Wa/Wb dilation. Use 3 with --quad-adapter-kernel-size 3 for ConvNeXt 7x7 layers.",
    )
    parser.add_argument("--adapter-scope", type=str, default="last_stage", choices=["last_stage", "all"])
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=5.0)

    parser.add_argument(
        "--print-trainable-names",
        action="store_true",
        help="Print names and sizes of all trainable parameter tensors.",
    )

    parser.add_argument(
        "--train-head-with-adapter",
        action="store_true",
        help="Train the classifier head together with LoRA/Quadratic adapters.",
    )

    parser.add_argument(
        "--save-non-linear-models",
        action="store_true",
        help=(
            "Also save model.pt for stages other than linear_base. "
            "By default, only linear_base saves model.pt; adapter/full/scratch "
            "runs save metrics.json only to save disk space."
        ),
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

    model = create_model(
        num_classes=len(class_names),
        device=device,
        stage=args.stage,
        base_checkpoint=base_checkpoint,
        adapter_rank=args.adapter_rank,
        adapter_alpha=args.adapter_alpha,
        quad_adapter_kernel_size=args.quad_adapter_kernel_size,
        quad_adapter_dilation=args.quad_adapter_dilation,
        adapter_scope=args.adapter_scope,
        train_head_with_adapter=args.train_head_with_adapter,
    )

    # Output dir
    output_dir = Path(args.output_dir) / args.dataset / args.stage  

    if args.stage in ("lora_dw", "quad_dw"):
        output_dir = output_dir / f"{args.adapter_scope}_rank_{args.adapter_rank}"
        if args.adapter_alpha is not None:
            safe_alpha = str(args.adapter_alpha).replace("-", "m").replace(".", "p")
            output_dir = output_dir / f"alpha_{safe_alpha}"
        if args.stage == "quad_dw" and args.quad_adapter_kernel_size is not None:
            output_dir = output_dir / f"quad_k{args.quad_adapter_kernel_size}_d{args.quad_adapter_dilation}"

    output_dir.mkdir(parents=True, exist_ok=True)

    report_params(model)
    if args.print_trainable_names:
        print_trainable_parameter_names(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    history = {
        "dataset": args.dataset,
        "stage": args.stage,
        "base_checkpoint": str(base_checkpoint) if base_checkpoint is not None else None,
        "quad_rank": args.quad_rank,
        "adapter_rank": args.adapter_rank,
        "adapter_alpha": args.adapter_alpha,
        "quad_adapter_kernel_size": args.quad_adapter_kernel_size,
        "quad_adapter_dilation": args.quad_adapter_dilation,
        "adapter_scope": args.adapter_scope,
        "train_head_with_adapter": args.train_head_with_adapter,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "trainable_params": count_trainable_params(model),
        "total_params": count_total_params(model),
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }

    start_time = time.time()

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device, args.grad_clip
        )
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

    if args.stage == "linear_base":
        save_checkpoint(
            model=model,
            classes=class_names,
            output_dir=output_dir,
            filename="model.pt",
            stage=args.stage,
            args_dict=args_dict,
            include_model_state=False,
        )
        print(f"Saved compact linear_base checkpoint to: {output_dir / 'model.pt'}")
    elif args.save_non_linear_models:
        save_checkpoint(
            model=model,
            classes=class_names,
            output_dir=output_dir,
            filename="model.pt",
            stage=args.stage,
            args_dict=args_dict,
            include_model_state=True,
        )
        print(f"Saved full non-linear-stage checkpoint to: {output_dir / 'model.pt'}")
    else:
        print(
            f"Skipping model.pt for stage='{args.stage}' to save disk space. "
            "metrics.json will still be saved."
        )

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
    
    

# 0) Train ResNet18 from scratch
# This starts from random weights and trains the full network.
# python train_LoRA_Qudratic.py \
#   --dataset cifar10 \
#   --stage scratch \
#   --epochs 50 \
#   --lr 1e-3

# 1) Train the frozen-backbone linear base
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

#python train_LoRA_Qudratic.py --dataset cifar10 --stage quadratic_adapter --base-checkpoint outputs/cifar10/linear_base/model.pt --quad-rank 4 --epochs 10 --lr 1e-3

# 3) Train only the LoRA adapter on top of that same frozen linear base
# If you omit --lora-rank, the script auto-picks one that roughly matches
# the quadratic head's trainable-parameter budget.
# python train_LoRA_Qudratic.py \
#   --dataset cifar10 \
#   --stage lora_adapter \
#   --base-checkpoint outputs/cifar10/linear_base/model.pt \
#   --quad-rank 4 \
#   --epochs 10 \
#   --lr 1e-3