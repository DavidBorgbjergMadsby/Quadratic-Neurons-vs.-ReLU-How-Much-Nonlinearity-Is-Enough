import math
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from helper_functions_updated import (
    set_seeds,
    plot_loss_curves,
    list_conv_layer_names,
    freeze_all_parameters,
    unfreeze_module,
    replace_conv_with_lora,
    print_trainable_parameters,
)


# ============================================================
# CONFIG
# ============================================================
ADAPTER_TYPE = "lora"      # options: "lora", "quadranet"

TRAIN_DIR = "data/train"
TEST_DIR = "data/test"

NUM_CLASSES = 3
BATCH_SIZE = 32
NUM_WORKERS = 2
EPOCHS = 10
LR = 1e-3

ADAPTER_R = 8
ADAPTER_ALPHA = 16.0
ADAPTER_DROPOUT = 0.1

TARGET_PREFIXES = ("features.6", "features.7")

SAVE_PATH = f"best_cnn_{ADAPTER_TYPE}_model.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# LoRA Conv2d
class LoRAConv2d(nn.Module):
    def __init__(self, conv, r=4, alpha=1.0, dropout=0.0):
        super().__init__()

        if conv.groups != 1:
            raise NotImplementedError("groups > 1 not supported")

        self.conv = conv
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.conv.weight.requires_grad = False
        if self.conv.bias is not None:
            self.conv.bias.requires_grad = False

        out_c, in_c, kh, kw = conv.weight.shape
        in_dim = in_c * kh * kw

        self.lora_A = nn.Parameter(torch.randn(r, in_dim) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_c, r))

    def forward(self, x):
        delta = (self.lora_B @ self.lora_A).view_as(self.conv.weight)
        weight = self.conv.weight + self.scaling * delta

        return nn.functional.conv2d(
            x,
            weight,
            bias=self.conv.bias,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
        )

# QuadraNetV2 Conv2d
class QuadraNetConv2d(nn.Module):
    def __init__(self, conv, r=4, alpha=1.0, dropout=0.0):
        super().__init__()

        if conv.groups != 1:
            raise NotImplementedError("groups > 1 not supported")

        self.conv = conv
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.conv.weight.requires_grad = False
        if self.conv.bias is not None:
            self.conv.bias.requires_grad = False

        out_c, in_c, kh, kw = conv.weight.shape
        self.out_c = out_c
        self.in_dim = in_c * kh * kw

        # Low-rank quadratic adapter:
        # x^T Wq x ≈ B(Ax)^2
        self.quad_A = nn.Parameter(torch.randn(r, self.in_dim) * 0.01)
        self.quad_B = nn.Parameter(torch.zeros(out_c, r))

    def forward(self, x):
        base_out = self.conv(x)

        patches = nn.functional.unfold(
            x,
            kernel_size=self.conv.kernel_size,
            dilation=self.conv.dilation,
            padding=self.conv.padding,
            stride=self.conv.stride,
        )

        patches = patches.transpose(1, 2)
        patches = self.dropout(patches)

        z = patches @ self.quad_A.T
        z2 = z ** 2
        quad_out = z2 @ self.quad_B.T

        batch_size = x.shape[0]
        out_h, out_w = base_out.shape[2], base_out.shape[3]

        quad_out = quad_out.transpose(1, 2).reshape(
            batch_size,
            self.out_c,
            out_h,
            out_w,
        )

        return base_out + self.scaling * quad_out

# ============================================================
# TRAIN / TEST
# ============================================================

def train_step(model, dataloader, loss_fn, optimizer):
    model.train()

    total_loss = 0
    total_acc = 0

    for X, y in dataloader:
        X, y = X.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()

        logits = model(X)
        loss = loss_fn(logits, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = logits.argmax(dim=1)
        total_acc += (preds == y).sum().item() / len(preds)

    return total_loss / len(dataloader), total_acc / len(dataloader)


@torch.inference_mode()
def test_step(model, dataloader, loss_fn):
    model.eval()

    total_loss = 0
    total_acc = 0

    for X, y in dataloader:
        X, y = X.to(DEVICE), y.to(DEVICE)

        logits = model(X)
        loss = loss_fn(logits, y)

        total_loss += loss.item()

        preds = logits.argmax(dim=1)
        total_acc += (preds == y).sum().item() / len(preds)

    return total_loss / len(dataloader), total_acc / len(dataloader)


# ============================================================
# MAIN
# ============================================================

def main():

    set_seeds(42)
    print("Using device:", DEVICE)

    # Load model
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)

    # Replace classifier
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, NUM_CLASSES),
    )

    # Freeze backbone
    freeze_all_parameters(model)
    unfreeze_module(model.classifier)

    # Select conv layers
    conv_names = list_conv_layer_names(model)

    target_names = {
        name for name in conv_names
        if name.startswith(TARGET_PREFIXES)
    }
    # Choose adapter
    if ADAPTER_TYPE == "lora":
        adapter_class = LoRAConv2d
    elif ADAPTER_TYPE == "quadranet":
        adapter_class = QuadraNetConv2d
    else:
        raise ValueError(f"Unknown ADAPTER_TYPE: {ADAPTER_TYPE}")

    # Apply adapter
    replace_conv_with_lora(
        model,
        lora_class=adapter_class,
        target_names=target_names,
        r=ADAPTER_R,
        alpha=ADAPTER_ALPHA,
        dropout=ADAPTER_DROPOUT,
    )

    model = model.to(DEVICE)

    print_trainable_parameters(model)

    # Data
    transforms = weights.transforms()

    train_data = datasets.ImageFolder(TRAIN_DIR, transform=transforms)
    test_data = datasets.ImageFolder(TEST_DIR, transform=transforms)

    train_loader = DataLoader(train_data, BATCH_SIZE, True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_data, BATCH_SIZE, False, num_workers=NUM_WORKERS)

    # Training setup
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR,
    )

    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }

    best_acc = 0
    best_weights = None

    # Training loop
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_step(model, train_loader, loss_fn, optimizer)
        test_loss, test_acc = test_step(model, test_loader, loss_fn)

        print(
            f"Epoch {epoch+1} | "
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
            f"test_loss={test_loss:.4f} | test_acc={test_acc:.4f}"
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        if test_acc > best_acc:
            best_acc = test_acc
            best_weights = model.state_dict()

    torch.save(best_weights, SAVE_PATH)
    print("Saved best model")

    plot_loss_curves(results)


# ============================================================

if __name__ == "__main__":
    main()