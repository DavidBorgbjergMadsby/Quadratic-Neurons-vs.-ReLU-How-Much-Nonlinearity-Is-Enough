import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("outputs/cifar10")

GATES = [0.0, 0.25, 0.5, 0.75, 1.0]

LORA_DIR = ROOT / "lora_adapter" / "rank_78_alpha_78"
QUAD_DIR = ROOT / "quadratic_adapter" / "rank_4"
LINEAR_DIR = ROOT / "linear_base"


def load_metrics(path):
    with open(path, "r") as f:
        return json.load(f)


rows = []

for gate in GATES:
    row = {"gate": gate}

    # ----------------------------
    # Linear
    # ----------------------------
    linear_file = LINEAR_DIR / f"relu_gate_{gate}" / "metrics.json"

    if linear_file.exists():
        data = load_metrics(linear_file)

        row["linear_train_acc"] = data["train_acc"][-1]
        row["linear_test_acc"] = data["test_acc"][-1]
        row["linear_time_sec"] = data["training_time_sec"]

    # ----------------------------
    # Quadratic
    # ----------------------------
    quad_file = QUAD_DIR / f"relu_gate_{gate}" / "metrics.json"

    if quad_file.exists():
        data = load_metrics(quad_file)

        row["quad_train_acc"] = data["train_acc"][-1]
        row["quad_test_acc"] = data["test_acc"][-1]
        row["quad_time_sec"] = data["training_time_sec"]

    # ----------------------------
    # LoRA
    # ----------------------------
    lora_file = LORA_DIR / f"relu_gate_{gate}" / "metrics.json"

    if lora_file.exists():
        data = load_metrics(lora_file)

        row["lora_train_acc"] = data["train_acc"][-1]
        row["lora_test_acc"] = data["test_acc"][-1]
        row["lora_time_sec"] = data["training_time_sec"]

    rows.append(row)

df = pd.DataFrame(rows)

# ==================================================
# Pretty table
# ==================================================

display_cols = [
    "gate",
    "linear_test_acc",
    "quad_test_acc",
    "lora_test_acc",
]

print("\n=== Experiment A: Test Accuracy Comparison ===\n")
print(df[display_cols].to_string(index=False))

# ==================================================
# Save CSV
# ==================================================

csv_file = "experimentA_comparison.csv"
df.to_csv(csv_file, index=False)

print(f"\nSaved CSV: {csv_file}")

# ==================================================
# Plot 1: Accuracy vs Gate
# ==================================================

plt.figure(figsize=(8, 5))

if "linear_test_acc" in df.columns:
    plt.plot(
        df["gate"],
        df["linear_test_acc"],
        marker="o",
        label="Linear Base",
    )

if "quad_test_acc" in df.columns:
    plt.plot(
        df["gate"],
        df["quad_test_acc"],
        marker="o",
        label="Quadratic Adapter",
    )

if "lora_test_acc" in df.columns:
    plt.plot(
        df["gate"],
        df["lora_test_acc"],
        marker="o",
        label="LoRA Adapter",
    )

plt.xlabel("ReLU Gate")
plt.ylabel("Test Accuracy")
plt.title("Accuracy vs ReLU Gate")
plt.grid(True)
plt.legend()

plt.savefig("accuracy_vs_gate.png", dpi=300, bbox_inches="tight")

plt.close()

print("Saved: accuracy_vs_gate.png")

# ==================================================
# Plot 2: Runtime vs Gate
# ==================================================

plt.figure(figsize=(8, 5))

if "linear_time_sec" in df.columns:
    plt.plot(
        df["gate"],
        df["linear_time_sec"],
        marker="o",
        label="Linear Base",
    )

if "quad_time_sec" in df.columns:
    plt.plot(
        df["gate"],
        df["quad_time_sec"],
        marker="o",
        label="Quadratic Adapter",
    )

if "lora_time_sec" in df.columns:
    plt.plot(
        df["gate"],
        df["lora_time_sec"],
        marker="o",
        label="LoRA Adapter",
    )

plt.xlabel("ReLU Gate")
plt.ylabel("Training Time (seconds)")
plt.title("Training Time vs ReLU Gate")
plt.grid(True)
plt.legend()

plt.savefig("runtime_vs_gate.png", dpi=300, bbox_inches="tight")

plt.close()

print("Saved: runtime_vs_gate.png")

# ==================================================
# Best gate per method
# ==================================================

print("\n=== Best Gates ===\n")

for method in [
    "linear_test_acc",
    "quad_test_acc",
    "lora_test_acc",
]:
    if method in df.columns:
        idx = df[method].idxmax()

        gate = df.loc[idx, "gate"]
        acc = df.loc[idx, method]

        print(f"{method}: best gate = {gate:.2f}, test_acc = {acc:.4f}")
