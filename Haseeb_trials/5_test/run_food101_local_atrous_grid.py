#!/usr/bin/env python3
"""
Local runner for the Food101 ConvNeXt LoRA / quadratic-adapter experiment.

This is a Python replacement for the HPC .sh sweep script. It runs the same
sequence locally by calling train_LoRA_Qudratic.py with subprocess.

Recommended placement:
    Put this file next to train_LoRA_Qudratic.py inside 5_test/

Typical usage from 5_test/:
    python run_food101_local_atrous_grid.py --seeds 45 --data-root ../4_test/data

For a quick check without training:
    python run_food101_local_atrous_grid.py --seeds 45 --dry-run
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence


@dataclass(frozen=True)
class AdapterRun:
    stage: str
    idx: int
    scope: str
    rank: int
    train_head: bool
    lr: str
    weight_decay: str
    quad_kernel: Optional[int] = None
    quad_dilation: Optional[int] = None
    adapter_alpha: Optional[str] = None


@dataclass(frozen=True)
class FullRun:
    idx: int
    lr: str
    weight_decay: str


def token(value: str) -> str:
    """Match the shell script's filename-safe LR/WD tokens."""
    return str(value).replace("-", "m").replace(".", "p")


def adapter_run_name(run: AdapterRun) -> str:
    prefix = "q" if run.stage == "quad_dw" else "l"
    stage_num = "1" if run.stage == "quad_dw" else "2"
    scope_tag = "all" if run.scope == "all" else "ls"
    head_tag = "h1" if run.train_head else "h0"
    name = (
        f"{stage_num}{run.idx:02d}_{prefix}_{scope_tag}_r{run.rank}_{head_tag}_"
        f"lr{token(run.lr)}_wd{token(run.weight_decay)}"
    )
    if run.stage == "quad_dw" and run.quad_kernel is not None and run.quad_dilation is not None:
        name += f"_k{run.quad_kernel}d{run.quad_dilation}"
    if run.adapter_alpha is not None:
        name += f"_a{token(run.adapter_alpha)}"
    return name


def full_run_name(run: FullRun) -> str:
    # Match the original shell script names exactly: 1e-4 -> 1e4, 3e-5 -> 3e5.
    lr_token = str(run.lr).replace("e-", "e").replace("-", "m").replace(".", "p")
    wd_token = token(run.weight_decay)
    return f"{run.idx:03d}_f_lr{lr_token}_wd{wd_token}"


def format_cmd(cmd: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(x)) for x in cmd)


def run_subprocess(cmd: Sequence[str], cwd: Path, dry_run: bool) -> None:
    print("\n$", format_cmd(cmd))
    if dry_run:
        return
    subprocess.run(list(map(str, cmd)), cwd=cwd, check=True)


def metrics_path_for(output_dir: Path, dataset: str, stage: str, run: Optional[AdapterRun] = None) -> Path:
    path = output_dir / dataset / stage
    if run is not None and stage in {"lora_dw", "quad_dw"}:
        path = path / f"{run.scope}_rank_{run.rank}"
        if run.adapter_alpha is not None:
            path = path / f"alpha_{token(run.adapter_alpha)}"
        if stage == "quad_dw" and run.quad_kernel is not None:
            path = path / f"quad_k{run.quad_kernel}_d{run.quad_dilation}"
    return path / "metrics.json"


def model_path_for(output_dir: Path, dataset: str, stage: str) -> Path:
    return output_dir / dataset / stage / "model.pt"


def should_skip_metrics(output_dir: Path, dataset: str, stage: str, resume: bool, run: Optional[AdapterRun] = None) -> bool:
    if not resume:
        return False
    return metrics_path_for(output_dir, dataset, stage, run).exists()


def write_manifest_row(writer: csv.writer, row: Sequence[object]) -> None:
    writer.writerow(row)


def build_train_command(
    python_exe: str,
    train_script: Path,
    dataset: str,
    data_root: Path,
    output_dir: Path,
    batch_size: int,
    num_workers: int,
    grad_clip: float,
    seed: int,
    stage: str,
    epochs: int,
    lr: str,
    weight_decay: str,
    print_trainable_names: bool,
    extra_args: Sequence[str] = (),
) -> List[str]:
    cmd = [
        python_exe,
        "-u",
        str(train_script),
        "--dataset",
        dataset,
        "--data-root",
        str(data_root),
        "--output-dir",
        str(output_dir),
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(num_workers),
        "--grad-clip",
        str(grad_clip),
        "--seed",
        str(seed),
        "--stage",
        stage,
        "--epochs",
        str(epochs),
        "--lr",
        str(lr),
        "--weight-decay",
        str(weight_decay),
    ]
    cmd.extend(extra_args)
    if print_trainable_names:
        cmd.append("--print-trainable-names")
    return cmd


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Food101 LoRA/quadratic adapter sweep locally.")
    parser.add_argument("--work-dir", type=Path, default=Path.cwd(), help="Folder where outputs/logs are saved. Default: current folder.")
    parser.add_argument("--data-root", type=Path, default=Path("../4_test/data"), help="Dataset root. From 5_test, use ../4_test/data.")
    parser.add_argument("--train-script", type=Path, default=Path("train_LoRA_Qudratic.py"), help="Path to train_LoRA_Qudratic.py.")
    parser.add_argument("--dataset", default="food101", choices=["cifar10", "cifar100", "flowers102", "pets", "food101"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[45], help="Seeds to run locally, for example: --seeds 45 or --seeds 21 45 69")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--linear-epochs", type=int, default=30)
    parser.add_argument("--adapter-epochs", type=int, default=16)
    parser.add_argument("--full-epochs", type=int, default=25)
    parser.add_argument("--linear-lr", default="1e-3")
    parser.add_argument("--linear-wd", default="0.05")
    parser.add_argument("--quad-adapter-kernel", type=int, default=3)
    parser.add_argument("--quad-adapter-dilation", type=int, default=3)
    parser.add_argument("--python", default=sys.executable, help="Python executable to use for training calls.")
    parser.add_argument("--sweep-name", default=None, help="Output sweep folder name. Default: food101_local_atrous_<timestamp>.")
    parser.add_argument("--resume", action="store_true", help="Skip a run if its metrics.json already exists.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument("--no-print-trainable-names", action="store_true", help="Do not pass --print-trainable-names to train script.")
    args = parser.parse_args(argv)

    work_dir = args.work_dir.expanduser().resolve()
    data_root = args.data_root.expanduser()
    if not data_root.is_absolute():
        data_root = (work_dir / data_root).resolve()
    train_script = args.train_script.expanduser()
    if not train_script.is_absolute():
        train_script = (work_dir / train_script).resolve()

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_name = args.sweep_name or f"food101_local_atrous_{timestamp}"
    sweep_root = work_dir / "outputs" / sweep_name
    manifest_path = sweep_root / "manifest.tsv"

    print("Work dir:    ", work_dir)
    print("Data root:   ", data_root)
    print("Train script:", train_script)
    print("Sweep root:  ", sweep_root)
    print("Seeds:       ", args.seeds)

    if not train_script.exists():
        raise FileNotFoundError(f"Could not find train script: {train_script}")
    if not data_root.exists():
        print(f"[warning] data root does not exist yet: {data_root}")
        print("          torchvision may download the dataset there if download=True.")

    quad_runs = [
        AdapterRun("quad_dw", 9, "all", 2, True, "1e-4", "0.0", args.quad_adapter_kernel, args.quad_adapter_dilation),
        AdapterRun("quad_dw", 10, "all", 2, True, "3e-4", "0.0", args.quad_adapter_kernel, args.quad_adapter_dilation),
        AdapterRun("quad_dw", 11, "all", 2, True, "3e-5", "0.0", args.quad_adapter_kernel, args.quad_adapter_dilation),
        AdapterRun("quad_dw", 12, "all", 1, True, "1e-4", "0.0", args.quad_adapter_kernel, args.quad_adapter_dilation),
        AdapterRun("quad_dw", 13, "all", 2, True, "1e-4", "0.05", args.quad_adapter_kernel, args.quad_adapter_dilation),
        AdapterRun("quad_dw", 14, "all", 2, True, "1e-4", "0.0", args.quad_adapter_kernel, args.quad_adapter_dilation, "0.5"),
        AdapterRun("quad_dw", 15, "all", 2, True, "1e-4", "0.0", args.quad_adapter_kernel, args.quad_adapter_dilation, "2.0"),
        AdapterRun("quad_dw", 16, "last_stage", 2, True, "1e-4", "0.0", args.quad_adapter_kernel, args.quad_adapter_dilation),
    ]
    lora_runs = [
        AdapterRun("lora_dw", 9, "all", 2, True, "1e-4", "0.0"),
        AdapterRun("lora_dw", 10, "all", 2, True, "3e-4", "0.0"),
        AdapterRun("lora_dw", 11, "all", 1, True, "1e-4", "0.0"),
        AdapterRun("lora_dw", 12, "all", 2, True, "1e-4", "0.05"),
        AdapterRun("lora_dw", 13, "last_stage", 2, True, "1e-4", "0.0"),
    ]
    full_runs = [
        FullRun(301, "1e-4", "0.01"),
        FullRun(302, "3e-5", "0.05"),
    ]

    sweep_root.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["seed", "run_name", "stage", "scope", "rank", "train_head", "lr", "weight_decay", "output_dir"])

        for seed in args.seeds:
            print("\n" + "#" * 60)
            print(f"STARTING SEED {seed}")
            print("#" * 60)

            seed_root = sweep_root / f"seed_{seed}"
            seed_root.mkdir(parents=True, exist_ok=True)

            linear_run = "000_lin_lr1e3_wd0p05"
            linear_out = seed_root / linear_run
            write_manifest_row(writer, [seed, linear_run, "linear_base", "none", "none", "head_only", args.linear_lr, args.linear_wd, linear_out])
            f.flush()

            if should_skip_metrics(linear_out, args.dataset, "linear_base", args.resume):
                print(f"[resume] skipping existing {linear_run}")
            else:
                cmd = build_train_command(
                    args.python,
                    train_script,
                    args.dataset,
                    data_root,
                    linear_out,
                    args.batch_size,
                    args.num_workers,
                    args.grad_clip,
                    seed,
                    "linear_base",
                    args.linear_epochs,
                    args.linear_lr,
                    args.linear_wd,
                    not args.no_print_trainable_names,
                )
                run_subprocess(cmd, cwd=work_dir, dry_run=args.dry_run)

            base_ckpt = model_path_for(linear_out, args.dataset, "linear_base")
            if not args.dry_run and not base_ckpt.exists():
                raise FileNotFoundError(f"Expected linear base checkpoint not found: {base_ckpt}")

            for run in quad_runs + lora_runs:
                run_name = adapter_run_name(run)
                out_dir = seed_root / run_name
                write_manifest_row(
                    writer,
                    [seed, run_name, run.stage, run.scope, run.rank, "yes" if run.train_head else "no", run.lr, run.weight_decay, out_dir],
                )
                f.flush()

                if should_skip_metrics(out_dir, args.dataset, run.stage, args.resume, run):
                    print(f"[resume] skipping existing {run_name}")
                    continue

                extra = [
                    "--base-checkpoint",
                    str(base_ckpt),
                    "--adapter-scope",
                    run.scope,
                    "--adapter-rank",
                    str(run.rank),
                ]
                if run.train_head:
                    extra.append("--train-head-with-adapter")
                if run.stage == "quad_dw" and run.quad_kernel is not None and run.quad_dilation is not None:
                    extra.extend(["--quad-adapter-kernel-size", str(run.quad_kernel), "--quad-adapter-dilation", str(run.quad_dilation)])
                if run.adapter_alpha is not None:
                    extra.extend(["--adapter-alpha", run.adapter_alpha])

                cmd = build_train_command(
                    args.python,
                    train_script,
                    args.dataset,
                    data_root,
                    out_dir,
                    args.batch_size,
                    args.num_workers,
                    args.grad_clip,
                    seed,
                    run.stage,
                    args.adapter_epochs,
                    run.lr,
                    run.weight_decay,
                    not args.no_print_trainable_names,
                    extra,
                )
                run_subprocess(cmd, cwd=work_dir, dry_run=args.dry_run)

            for run in full_runs:
                run_name = full_run_name(run)
                out_dir = seed_root / run_name
                write_manifest_row(writer, [seed, run_name, "full_finetune", "all", "full", "full", run.lr, run.weight_decay, out_dir])
                f.flush()

                if should_skip_metrics(out_dir, args.dataset, "full_finetune", args.resume):
                    print(f"[resume] skipping existing {run_name}")
                    continue

                cmd = build_train_command(
                    args.python,
                    train_script,
                    args.dataset,
                    data_root,
                    out_dir,
                    args.batch_size,
                    args.num_workers,
                    args.grad_clip,
                    seed,
                    "full_finetune",
                    args.full_epochs,
                    run.lr,
                    run.weight_decay,
                    not args.no_print_trainable_names,
                )
                run_subprocess(cmd, cwd=work_dir, dry_run=args.dry_run)

    print("\nDone.")
    print(f"Manifest: {manifest_path}")
    print(f"Outputs:  {sweep_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
