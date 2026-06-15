#!/usr/bin/env python3
"""
Local runner for the Food101 high-rank atrous Quadratic vs LoRA sweep.

This is the local Python equivalent of the LSF/bash sweep:
  1) Train a linear_base checkpoint.
  2) Train high-rank atrous Quadratic adapters from that checkpoint.
  3) Train matched high-rank LoRA adapters from that checkpoint.

Example:
  python run_food101_local_high_rank.py \
    --train-script train_LoRA_Qudratic.py \
    --data-root ~/Desktop/Fagprojekt/4_test/data \
    --output-root ./outputs/food101_local_atrous_budget_scaling \
    --seeds 21 \
    --batch-size 32 \
    --num-workers 2

Preview commands without running:
  python run_food101_local_high_rank.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import platform
import shlex
import subprocess
import sys
from pathlib import Path


def expand_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def safe_float_tag(value: str | float) -> str:
    return str(value).replace("-", "m").replace(".", "p")


def lr_to_name(lr: str | float) -> str:
    # Examples: 1e-4 -> 1em4, 3e-4 -> 3em4
    return str(lr).replace("-", "m").replace(".", "p")


def make_run_name(
    stage: str,
    idx: int,
    scope: str,
    rank: int,
    train_head: bool,
    lr: str,
    wd: str,
    quad_kernel: int | None = None,
    quad_dilation: int | None = None,
    adapter_alpha: str | None = None,
) -> str:
    prefix = "q" if stage == "quad_dw" else "l"
    stage_num = "1" if stage == "quad_dw" else "2"
    scope_tag = "all" if scope == "all" else "ls"
    htag = "h1" if train_head else "h0"

    name = (
        f"{stage_num}{idx:02d}_{prefix}_{scope_tag}_r{rank}_{htag}_"
        f"lr{lr_to_name(lr)}_wd{safe_float_tag(wd)}"
    )

    if stage == "quad_dw" and quad_kernel is not None and quad_dilation is not None:
        name += f"_k{quad_kernel}d{quad_dilation}"

    if adapter_alpha is not None:
        name += f"_a{safe_float_tag(adapter_alpha)}"

    return name


def run_command(cmd: list[str], dry_run: bool = False) -> None:
    printable = " ".join(shlex.quote(part) for part in cmd)
    print("\n$ " + printable, flush=True)
    if dry_run:
        return

    completed = subprocess.run(cmd)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {printable}")


def write_manifest_header(manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        f.write("seed\trun_name\tstage\tscope\trank\ttrain_head\tlr\tweight_decay\toutput_dir\n")


def append_manifest(
    manifest_path: Path,
    seed: int,
    run_name: str,
    stage: str,
    scope: str,
    rank: str | int,
    train_head: str,
    lr: str,
    wd: str,
    output_dir: Path,
) -> None:
    with manifest_path.open("a", encoding="utf-8") as f:
        f.write(
            f"{seed}\t{run_name}\t{stage}\t{scope}\t{rank}\t{train_head}\t"
            f"{lr}\t{wd}\t{output_dir}\n"
        )


def run_train(
    *,
    python_exe: str,
    train_script: Path,
    dataset: str,
    data_root: Path,
    sweep_root: Path,
    seed: int,
    run_name: str,
    stage_args: list[str],
    batch_size: int,
    num_workers: int,
    grad_clip: float,
    print_trainable_names: bool,
    dry_run: bool,
) -> Path:
    output_dir = sweep_root / f"seed_{seed}" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"SEED {seed} | RUN {run_name}")
    print(f"Output dir: {output_dir}")
    print("=" * 60)

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
        *stage_args,
    ]

    if print_trainable_names:
        cmd.append("--print-trainable-names")

    run_command(cmd, dry_run=dry_run)
    return output_dir


def run_adapter(
    *,
    python_exe: str,
    train_script: Path,
    dataset: str,
    data_root: Path,
    sweep_root: Path,
    manifest_path: Path,
    seed: int,
    stage: str,
    idx: int,
    scope: str,
    rank: int,
    train_head: bool,
    lr: str,
    wd: str,
    base_ckpt: Path,
    epochs: int,
    batch_size: int,
    num_workers: int,
    grad_clip: float,
    print_trainable_names: bool,
    dry_run: bool,
    quad_kernel: int | None = None,
    quad_dilation: int | None = None,
    adapter_alpha: str | None = None,
) -> None:
    if stage not in {"quad_dw", "lora_dw"}:
        raise ValueError("stage must be either 'quad_dw' or 'lora_dw'")

    run_name = make_run_name(
        stage=stage,
        idx=idx,
        scope=scope,
        rank=rank,
        train_head=train_head,
        lr=lr,
        wd=wd,
        quad_kernel=quad_kernel,
        quad_dilation=quad_dilation,
        adapter_alpha=adapter_alpha,
    )

    output_dir = sweep_root / f"seed_{seed}" / run_name
    append_manifest(
        manifest_path=manifest_path,
        seed=seed,
        run_name=run_name,
        stage=stage,
        scope=scope,
        rank=rank,
        train_head="yes" if train_head else "no",
        lr=lr,
        wd=wd,
        output_dir=output_dir,
    )

    stage_args = [
        "--stage",
        stage,
        "--base-checkpoint",
        str(base_ckpt),
        "--adapter-scope",
        scope,
        "--adapter-rank",
        str(rank),
        "--epochs",
        str(epochs),
        "--lr",
        str(lr),
        "--weight-decay",
        str(wd),
    ]

    if train_head:
        stage_args.append("--train-head-with-adapter")

    if stage == "quad_dw":
        if (quad_kernel is None) != (quad_dilation is None):
            raise ValueError("quad_kernel and quad_dilation must be given together")
        if quad_kernel is not None and quad_dilation is not None:
            stage_args.extend(
                [
                    "--quad-adapter-kernel-size",
                    str(quad_kernel),
                    "--quad-adapter-dilation",
                    str(quad_dilation),
                ]
            )

    if adapter_alpha is not None:
        stage_args.extend(["--adapter-alpha", str(adapter_alpha)])

    run_train(
        python_exe=python_exe,
        train_script=train_script,
        dataset=dataset,
        data_root=data_root,
        sweep_root=sweep_root,
        seed=seed,
        run_name=run_name,
        stage_args=stage_args,
        batch_size=batch_size,
        num_workers=num_workers,
        grad_clip=grad_clip,
        print_trainable_names=print_trainable_names,
        dry_run=dry_run,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-script", default="train_LoRA_Qudratic.py")
    parser.add_argument("--dataset", default="food101")
    parser.add_argument("--data-root", default="~/Desktop/Fagprojekt/4_test/data")
    parser.add_argument("--output-root", default="./outputs/food101_local_atrous_budget_scaling")
    parser.add_argument("--seeds", nargs="+", type=int, default=[21])
    parser.add_argument("--linear-epochs", type=int, default=30)
    parser.add_argument("--adapter-epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--linear-lr", default="1e-3")
    parser.add_argument("--linear-wd", default="0.05")
    parser.add_argument("--quad-adapter-kernel", type=int, default=3)
    parser.add_argument("--quad-adapter-dilation", type=int, default=3)
    parser.add_argument("--print-trainable-names", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-linear-base-if-exists",
        action="store_true",
        help="Reuse the expected linear_base checkpoint if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    python_exe = sys.executable
    train_script = expand_path(args.train_script)
    data_root = expand_path(args.data_root)
    sweep_root = expand_path(args.output_root)
    manifest_path = sweep_root / "manifest.tsv"

    if not train_script.exists():
        raise FileNotFoundError(
            f"Training script not found: {train_script}\n"
            "Run from the folder containing train_LoRA_Qudratic.py or pass --train-script."
        )

    sweep_root.mkdir(parents=True, exist_ok=True)
    write_manifest_header(manifest_path)

    os.environ.setdefault("TORCH_HOME", str((Path.cwd() / "torch_cache").resolve()))
    Path(os.environ["TORCH_HOME"]).mkdir(parents=True, exist_ok=True)

    print("Local high-rank Food101 sweep")
    print(f"Host: {platform.node()}")
    print(f"Python: {python_exe}")
    print(f"Training script: {train_script}")
    print(f"Dataset: {args.dataset}")
    print(f"Data root: {data_root}")
    print(f"Output root: {sweep_root}")
    print(f"Manifest: {manifest_path}")
    print(f"Dry run: {args.dry_run}")

    quad_runs = [
        # idx, scope, rank, train_head, lr, wd
        (17, "all", 4, True, "1e-4", "0.0"),
        (18, "all", 4, True, "3e-4", "0.0"),
        (19, "all", 8, True, "1e-4", "0.0"),
        (20, "all", 8, True, "3e-4", "0.0"),
    ]

    lora_runs = [
        # idx, scope, rank, train_head, lr, wd
        (14, "all", 4, True, "1e-4", "0.0"),
        (15, "all", 4, True, "3e-4", "0.0"),
        (16, "all", 8, True, "1e-4", "0.0"),
        (17, "all", 8, True, "3e-4", "0.0"),
    ]

    for seed in args.seeds:
        print("#" * 60)
        print(f"STARTING SEED {seed}")
        print("#" * 60)

        seed_root = sweep_root / f"seed_{seed}"
        linear_run = "000_lin_lr1e3_wd0p05"
        linear_output_dir = seed_root / linear_run
        expected_base_ckpt = linear_output_dir / args.dataset / "linear_base" / "model.pt"

        append_manifest(
            manifest_path=manifest_path,
            seed=seed,
            run_name=linear_run,
            stage="linear_base",
            scope="none",
            rank="none",
            train_head="head_only",
            lr=args.linear_lr,
            wd=args.linear_wd,
            output_dir=linear_output_dir,
        )

        if args.skip_linear_base_if_exists and expected_base_ckpt.exists():
            print(f"Reusing existing base checkpoint: {expected_base_ckpt}")
        else:
            run_train(
                python_exe=python_exe,
                train_script=train_script,
                dataset=args.dataset,
                data_root=data_root,
                sweep_root=sweep_root,
                seed=seed,
                run_name=linear_run,
                stage_args=[
                    "--stage",
                    "linear_base",
                    "--epochs",
                    str(args.linear_epochs),
                    "--lr",
                    str(args.linear_lr),
                    "--weight-decay",
                    str(args.linear_wd),
                ],
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                grad_clip=args.grad_clip,
                print_trainable_names=args.print_trainable_names,
                dry_run=args.dry_run,
            )

        base_ckpt = expected_base_ckpt
        if not args.dry_run and not base_ckpt.exists():
            raise FileNotFoundError(f"Expected linear base checkpoint not found: {base_ckpt}")

        print(f"Using base checkpoint for adapters: {base_ckpt}")

        for idx, scope, rank, train_head, lr, wd in quad_runs:
            run_adapter(
                python_exe=python_exe,
                train_script=train_script,
                dataset=args.dataset,
                data_root=data_root,
                sweep_root=sweep_root,
                manifest_path=manifest_path,
                seed=seed,
                stage="quad_dw",
                idx=idx,
                scope=scope,
                rank=rank,
                train_head=train_head,
                lr=lr,
                wd=wd,
                base_ckpt=base_ckpt,
                epochs=args.adapter_epochs,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                grad_clip=args.grad_clip,
                print_trainable_names=args.print_trainable_names,
                dry_run=args.dry_run,
                quad_kernel=args.quad_adapter_kernel,
                quad_dilation=args.quad_adapter_dilation,
            )

        for idx, scope, rank, train_head, lr, wd in lora_runs:
            run_adapter(
                python_exe=python_exe,
                train_script=train_script,
                dataset=args.dataset,
                data_root=data_root,
                sweep_root=sweep_root,
                manifest_path=manifest_path,
                seed=seed,
                stage="lora_dw",
                idx=idx,
                scope=scope,
                rank=rank,
                train_head=train_head,
                lr=lr,
                wd=wd,
                base_ckpt=base_ckpt,
                epochs=args.adapter_epochs,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                grad_clip=args.grad_clip,
                print_trainable_names=args.print_trainable_names,
                dry_run=args.dry_run,
            )

        print("#" * 60)
        print(f"FINISHED SEED {seed}")
        print("#" * 60)

    print("All seeds finished.")
    print(f"Sweep root: {sweep_root}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
