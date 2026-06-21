#!/usr/bin/env python3
"""
Local Food101 ConvNeXt-Tiny experiment runner.

Runs, in order:
  1 linear_base
  8 quad_dw runs
  8 lora_dw runs
  2 full_finetune runs

Each experiment gets its own output folder, log, command file, metrics.json,
and best_model.pt. The script creates a patched copy of train_LoRA_Qudratic.py
that saves best_model.pt whenever test_acc improves.

Example:
  python run_food101_local_8q_8l_2f.py

Useful local overrides:
  python run_food101_local_8q_8l_2f.py --batch-size 32 --num-workers 4
  python run_food101_local_8q_8l_2f.py --adapter-epochs 5 --full-epochs 5
  python run_food101_local_8q_8l_2f.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class AdapterCombo:
    scope: str
    rank: int
    train_head: bool
    lr: str
    weight_decay: str


@dataclass(frozen=True)
class FullCombo:
    lr: str
    weight_decay: str


QUAD_COMBOS: Tuple[AdapterCombo, ...] = (
    AdapterCombo("last_stage", 1, False, "1e-3", "0.0"),
    AdapterCombo("last_stage", 2, False, "1e-3", "0.0"),
    AdapterCombo("last_stage", 4, False, "1e-3", "0.0"),
    AdapterCombo("last_stage", 2, False, "3e-4", "0.01"),
    AdapterCombo("all",        1, False, "3e-4", "0.0"),
    AdapterCombo("all",        2, False, "3e-4", "0.0"),
    AdapterCombo("last_stage", 2, True,  "1e-4", "0.0"),
    AdapterCombo("all",        2, True,  "1e-4", "0.0"),
)

LORA_COMBOS: Tuple[AdapterCombo, ...] = QUAD_COMBOS

FULL_COMBOS: Tuple[FullCombo, ...] = (
    FullCombo("1e-4", "0.01"),
    FullCombo("3e-5", "0.05"),
)


def sanitize(value: object) -> str:
    s = str(value)
    return (
        s.replace("+", "")
        .replace("-", "m")
        .replace(".", "p")
        .replace("=", "")
        .replace(" ", "_")
    )


def quote_cmd(cmd: Sequence[str]) -> str:
    """Cross-platform readable command string."""
    if os.name == "nt":
        return subprocess.list2cmdline(list(cmd))
    return " ".join(shlex.quote(str(x)) for x in cmd)


def mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def patch_training_script(src: Path, dst: Path) -> None:
    """Create a patched train script that saves best_model.pt by best test_acc."""
    if not src.exists():
        raise FileNotFoundError(f"Training script not found: {src}")

    text = src.read_text(encoding="utf-8")

    if "filename=\"best_model.pt\"" in text or "best_test_acc = -float(\"inf\")" in text:
        # Already patched enough; copy as-is to the requested dst.
        dst.write_text(text, encoding="utf-8")
        return

    anchor_start = "    start_time = time.time()\n\n    for epoch in range(args.epochs):\n"
    replacement_start = (
        "    start_time = time.time()\n"
        "    best_test_acc = -float(\"inf\")\n"
        "    best_epoch = -1\n\n"
        "    for epoch in range(args.epochs):\n"
    )
    if anchor_start not in text:
        raise RuntimeError(
            "Could not patch training script: did not find the epoch-loop start anchor. "
            "Look for `start_time = time.time()` followed by `for epoch in range(args.epochs):`."
        )
    text = text.replace(anchor_start, replacement_start, 1)

    anchor_after_eval = "        history[\"test_acc\"].append(test_acc)\n\n        print(\n"
    replacement_after_eval = (
        "        history[\"test_acc\"].append(test_acc)\n\n"
        "        if test_acc > best_test_acc:\n"
        "            best_test_acc = test_acc\n"
        "            best_epoch = epoch + 1\n"
        "            args_dict_best = dict(vars(args))\n"
        "            if args_dict_best[\"base_checkpoint\"] is not None:\n"
        "                args_dict_best[\"base_checkpoint\"] = str(args_dict_best[\"base_checkpoint\"])\n"
        "            save_checkpoint(\n"
        "                model=model,\n"
        "                classes=class_names,\n"
        "                output_dir=output_dir,\n"
        "                filename=\"best_model.pt\",\n"
        "                stage=args.stage,\n"
        "                args_dict=args_dict_best,\n"
        "            )\n"
        "            with open(output_dir / \"best_so_far.json\", \"w\") as f:\n"
        "                json.dump({\n"
        "                    \"best_epoch\": best_epoch,\n"
        "                    \"best_test_acc\": best_test_acc,\n"
        "                    \"test_loss_at_best\": test_loss,\n"
        "                    \"train_acc_at_best\": train_acc,\n"
        "                    \"train_loss_at_best\": train_loss,\n"
        "                }, f, indent=2)\n"
        "            print(f\"[best] epoch {best_epoch} | test_acc: {best_test_acc:.4f}\")\n\n"
        "        print(\n"
    )
    if anchor_after_eval not in text:
        raise RuntimeError(
            "Could not patch training script: did not find the post-evaluation history anchor. "
            "Look for `history[\"test_acc\"].append(test_acc)` followed by `print(`."
        )
    text = text.replace(anchor_after_eval, replacement_after_eval, 1)

    anchor_history_time = "    history[\"training_time_sec\"] = elapsed\n"
    replacement_history_time = (
        "    history[\"training_time_sec\"] = elapsed\n"
        "    history[\"best_epoch\"] = best_epoch\n"
        "    history[\"best_test_acc\"] = best_test_acc\n"
    )
    if anchor_history_time not in text:
        raise RuntimeError(
            "Could not patch training script: did not find `history[\"training_time_sec\"] = elapsed`."
        )
    text = text.replace(anchor_history_time, replacement_history_time, 1)

    dst.write_text(text, encoding="utf-8")


def run_and_tee(cmd: Sequence[str], log_path: Path, cwd: Path, env: Optional[dict] = None) -> int:
    """Run command, stream combined stdout/stderr to console and log file."""
    mkdir(log_path.parent)
    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        log_file.write(f"Command: {quote_cmd(cmd)}\n")
        log_file.write(f"CWD: {cwd}\n")
        log_file.write(f"Started: {dt.datetime.now().isoformat(timespec='seconds')}\n")
        log_file.write("=" * 80 + "\n")
        log_file.flush()

        proc = subprocess.Popen(
            list(cmd),
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log_file.write(line)
        proc.wait()
        log_file.write("=" * 80 + "\n")
        log_file.write(f"Finished: {dt.datetime.now().isoformat(timespec='seconds')}\n")
        log_file.write(f"Exit code: {proc.returncode}\n")
        return int(proc.returncode)


def write_manifest_header(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write("run_name\tstage\tscope\trank\thead\tlr\tweight_decay\toutput_dir\n")


def append_manifest(
    path: Path,
    run_name: str,
    stage: str,
    scope: str,
    rank: str,
    head: str,
    lr: str,
    weight_decay: str,
    output_dir: Path,
) -> None:
    with path.open("a", encoding="utf-8", newline="") as f:
        f.write(
            f"{run_name}\t{stage}\t{scope}\t{rank}\t{head}\t{lr}\t{weight_decay}\t{output_dir}\n"
        )


def build_base_args(args: argparse.Namespace, run_dir: Path) -> List[str]:
    cmd = [
        sys.executable,
        str(args.train_runner),
        "--dataset", args.dataset,
        "--data-root", str(args.data_root),
        "--output-dir", str(run_dir),
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--grad-clip", str(args.grad_clip),
        "--seed", str(args.seed),
    ]
    if args.print_trainable_names:
        cmd.append("--print-trainable-names")
    return cmd


def run_exp(
    *,
    args: argparse.Namespace,
    run_index: int,
    run_name: str,
    metrics_rel: Path,
    train_args: Sequence[str],
    expected_runs: int,
) -> bool:
    run_dir = args.sweep_root / run_name
    metrics_path = run_dir / metrics_rel
    done_file = run_dir / "DONE"
    failed_file = run_dir / "FAILED"
    command_file = run_dir / "command.txt"
    log_file = run_dir / "run.log"
    mkdir(run_dir)

    if args.skip_existing and done_file.exists() and metrics_path.exists():
        print(f"[{run_index}/{expected_runs}] SKIP existing run: {run_name}")
        return True

    if failed_file.exists():
        failed_file.unlink()

    full_cmd = build_base_args(args, run_dir) + list(train_args)
    command_file.write_text(quote_cmd(full_cmd) + "\n", encoding="utf-8")

    print("=" * 80)
    print(f"[{run_index}/{expected_runs}] RUN: {run_name}")
    print(f"Output: {run_dir}")
    print(f"Expected metrics: {metrics_path}")
    print(f"Command: {quote_cmd(full_cmd)}")
    print("=" * 80)

    if args.dry_run:
        return True

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("OMP_NUM_THREADS", str(args.omp_num_threads))
    env.setdefault("MKL_NUM_THREADS", str(args.mkl_num_threads))
    if args.torch_home:
        env["TORCH_HOME"] = str(args.torch_home)

    status = run_and_tee(full_cmd, log_file, cwd=args.project_dir, env=env)
    if status != 0:
        failed_file.write_text(f"Run failed with exit code {status}\n", encoding="utf-8")
        print(f"ERROR: run failed with exit code {status}: {run_name}")
        if not args.continue_on_fail:
            raise SystemExit(status)
        return False

    if not metrics_path.exists():
        failed_file.write_text(f"Metrics file missing: {metrics_path}\n", encoding="utf-8")
        print(f"ERROR: metrics file missing after successful run: {metrics_path}")
        if not args.continue_on_fail:
            raise SystemExit(1)
        return False

    done_file.write_text(dt.datetime.now().isoformat(timespec="seconds") + "\n", encoding="utf-8")
    print(f"Finished run: {run_name}")
    return True


def best_checkpoint_from_linear(args: argparse.Namespace, linear_run_name: str) -> Path:
    base_dir = args.sweep_root / linear_run_name / args.dataset / "linear_base"
    candidates = [base_dir / "best_model.pt", base_dir / "model.pt"]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Expected linear checkpoint not found in {base_dir}")


def build_summary(root: Path, summary_csv: Path) -> None:
    rows = []
    for metrics_path in root.rglob("metrics.json"):
        try:
            data = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"Could not read {metrics_path}: {exc}")
            continue

        test_acc = data.get("test_acc", []) or []
        test_loss = data.get("test_loss", []) or []
        train_acc = data.get("train_acc", []) or []
        train_loss = data.get("train_loss", []) or []

        if test_acc:
            best_i = max(range(len(test_acc)), key=lambda i: test_acc[i])
            best_epoch = best_i + 1
            best_test_acc = test_acc[best_i]
            best_test_loss = test_loss[best_i] if best_i < len(test_loss) else ""
            train_acc_at_best = train_acc[best_i] if best_i < len(train_acc) else ""
            train_loss_at_best = train_loss[best_i] if best_i < len(train_loss) else ""
            last_test_acc = test_acc[-1]
        else:
            best_epoch = data.get("best_epoch", "")
            best_test_acc = data.get("best_test_acc", "")
            best_test_loss = ""
            train_acc_at_best = ""
            train_loss_at_best = ""
            last_test_acc = ""

        rel = metrics_path.relative_to(root)
        run_name = rel.parts[0] if rel.parts else metrics_path.parent.name
        rows.append(
            {
                "run_name": run_name,
                "stage": data.get("stage", ""),
                "best_epoch": best_epoch,
                "best_test_acc": best_test_acc,
                "best_test_loss": best_test_loss,
                "train_acc_at_best": train_acc_at_best,
                "train_loss_at_best": train_loss_at_best,
                "last_test_acc": last_test_acc,
                "trainable_params": data.get("trainable_params", ""),
                "total_params": data.get("total_params", ""),
                "training_time_sec": data.get("training_time_sec", ""),
                "metrics_path": str(rel),
            }
        )

    def sort_key(row: dict) -> float:
        try:
            return float(row["best_test_acc"])
        except Exception:
            return -1.0

    rows.sort(key=sort_key, reverse=True)
    mkdir(summary_csv.parent)
    fieldnames = [
        "run_name",
        "stage",
        "best_epoch",
        "best_test_acc",
        "best_test_loss",
        "train_acc_at_best",
        "train_loss_at_best",
        "last_test_acc",
        "trainable_params",
        "total_params",
        "training_time_sec",
        "metrics_path",
    ]
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote summary: {summary_csv}")
    print("Top runs:")
    for row in rows[:20]:
        print(
            f"{row['best_test_acc']}\tepoch {row['best_epoch']}\t"
            f"{row['stage']}\t{row['run_name']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local Food101 sweep: 1 linear, 8 quad, 8 LoRA, 2 full fine-tune."
    )
    parser.add_argument("--project-dir", type=Path, default=Path.cwd(), help="Folder containing train_LoRA_Qudratic.py")
    parser.add_argument("--train-script", type=Path, default=Path("train_LoRA_Qudratic.py"))
    parser.add_argument("--patched-script-name", type=str, default="train_LoRA_Qudratic_local_best.py")
    parser.add_argument("--dataset", type=str, default="food101")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--run-stamp", type=str, default=None)
    parser.add_argument("--torch-home", type=Path, default=Path("torch_cache"))

    parser.add_argument("--batch-size", type=int, default=32, help="Local default is 32; increase if your GPU can handle it.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--omp-num-threads", type=int, default=4)
    parser.add_argument("--mkl-num-threads", type=int, default=4)

    parser.add_argument("--linear-epochs", type=int, default=30)
    parser.add_argument("--adapter-epochs", type=int, default=12)
    parser.add_argument("--full-epochs", type=int, default=25)

    parser.add_argument("--linear-lr", type=str, default="1e-3")
    parser.add_argument("--linear-weight-decay", type=str, default="0.05")

    parser.add_argument("--continue-on-fail", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--patch-best", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--print-trainable-names", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    args.project_dir = args.project_dir.expanduser().resolve()
    if not args.train_script.is_absolute():
        args.train_script = args.project_dir / args.train_script
    else:
        args.train_script = args.train_script.resolve()

    if not args.data_root.is_absolute():
        args.data_root = args.project_dir / args.data_root
    if not args.output_root.is_absolute():
        args.output_root = args.project_dir / args.output_root
    if args.torch_home and not args.torch_home.is_absolute():
        args.torch_home = args.project_dir / args.torch_home

    if args.run_stamp is None:
        args.run_stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    args.sweep_root = args.output_root / f"{args.dataset}_local_8q8l2f_{args.run_stamp}"
    args.train_runner = args.project_dir / args.patched_script_name if args.patch_best else args.train_script

    return args


def main() -> None:
    args = parse_args()
    mkdir(args.sweep_root)
    mkdir(args.data_root)
    mkdir(args.output_root)
    if args.torch_home:
        mkdir(args.torch_home)

    print("========== Local sweep info ==========")
    print(f"Date: {dt.datetime.now().isoformat(timespec='seconds')}")
    print(f"Platform: {platform.platform()}")
    print(f"Python executable: {sys.executable}")
    print(f"Project dir: {args.project_dir}")
    print(f"Train script: {args.train_script}")
    print(f"Train runner: {args.train_runner}")
    print(f"Sweep root: {args.sweep_root}")
    print(f"Dataset: {args.dataset}")
    print(f"Batch size: {args.batch_size}")

    if args.patch_best:
        print(f"Creating patched training script with best_model.pt saving: {args.train_runner}")
        patch_training_script(args.train_script, args.train_runner)
        compile_cmd = [sys.executable, "-m", "py_compile", str(args.train_runner)]
        status = subprocess.call(compile_cmd, cwd=str(args.project_dir))
        if status != 0:
            raise SystemExit(status)

    manifest = args.sweep_root / "manifest.tsv"
    summary_csv = args.sweep_root / "summary.csv"
    write_manifest_header(manifest)

    total_expected = 1 + len(QUAD_COMBOS) + len(LORA_COMBOS) + len(FULL_COMBOS)
    run_index = 0
    fail_count = 0

    # 1) Linear base.
    linear_run = f"000_linear_base_lr_{sanitize(args.linear_lr)}_wd_{sanitize(args.linear_weight_decay)}"
    append_manifest(
        manifest,
        linear_run,
        "linear_base",
        "none",
        "none",
        "head_only",
        args.linear_lr,
        args.linear_weight_decay,
        args.sweep_root / linear_run,
    )
    run_index += 1
    ok = run_exp(
        args=args,
        run_index=run_index,
        expected_runs=total_expected,
        run_name=linear_run,
        metrics_rel=Path(args.dataset) / "linear_base" / "metrics.json",
        train_args=[
            "--stage", "linear_base",
            "--epochs", str(args.linear_epochs),
            "--lr", args.linear_lr,
            "--weight-decay", args.linear_weight_decay,
        ],
    )
    fail_count += 0 if ok else 1

    if args.dry_run:
        base_ckpt = args.sweep_root / linear_run / args.dataset / "linear_base" / "best_model.pt"
        print(f"Dry-run base checkpoint placeholder: {base_ckpt}")
    else:
        base_ckpt = best_checkpoint_from_linear(args, linear_run)
        print(f"Using base checkpoint for adapters: {base_ckpt}")

    # 2) 8 quadratic runs.
    print("========== Starting exactly 8 quad_dw runs ==========")
    for i, combo in enumerate(QUAD_COMBOS, start=1):
        head = "yes" if combo.train_head else "no"
        run_name = (
            f"1{i:02d}_quad_dw_{combo.scope}_r{combo.rank}_{head}head_"
            f"lr_{sanitize(combo.lr)}_wd_{sanitize(combo.weight_decay)}"
        )
        append_manifest(
            manifest,
            run_name,
            "quad_dw",
            combo.scope,
            str(combo.rank),
            head,
            combo.lr,
            combo.weight_decay,
            args.sweep_root / run_name,
        )
        train_args = [
            "--stage", "quad_dw",
            "--base-checkpoint", str(base_ckpt),
            "--adapter-scope", combo.scope,
            "--adapter-rank", str(combo.rank),
            "--epochs", str(args.adapter_epochs),
            "--lr", combo.lr,
            "--weight-decay", combo.weight_decay,
        ]
        if combo.train_head:
            train_args.append("--train-head-with-adapter")
        run_index += 1
        ok = run_exp(
            args=args,
            run_index=run_index,
            expected_runs=total_expected,
            run_name=run_name,
            metrics_rel=Path(args.dataset) / "quad_dw" / f"{combo.scope}_rank_{combo.rank}" / "metrics.json",
            train_args=train_args,
        )
        fail_count += 0 if ok else 1

    # 3) 8 LoRA runs.
    print("========== Starting exactly 8 lora_dw runs ==========")
    for i, combo in enumerate(LORA_COMBOS, start=1):
        head = "yes" if combo.train_head else "no"
        run_name = (
            f"2{i:02d}_lora_dw_{combo.scope}_r{combo.rank}_{head}head_"
            f"lr_{sanitize(combo.lr)}_wd_{sanitize(combo.weight_decay)}"
        )
        append_manifest(
            manifest,
            run_name,
            "lora_dw",
            combo.scope,
            str(combo.rank),
            head,
            combo.lr,
            combo.weight_decay,
            args.sweep_root / run_name,
        )
        train_args = [
            "--stage", "lora_dw",
            "--base-checkpoint", str(base_ckpt),
            "--adapter-scope", combo.scope,
            "--adapter-rank", str(combo.rank),
            "--epochs", str(args.adapter_epochs),
            "--lr", combo.lr,
            "--weight-decay", combo.weight_decay,
        ]
        if combo.train_head:
            train_args.append("--train-head-with-adapter")
        run_index += 1
        ok = run_exp(
            args=args,
            run_index=run_index,
            expected_runs=total_expected,
            run_name=run_name,
            metrics_rel=Path(args.dataset) / "lora_dw" / f"{combo.scope}_rank_{combo.rank}" / "metrics.json",
            train_args=train_args,
        )
        fail_count += 0 if ok else 1

    # 4) 2 full fine-tune runs.
    print("========== Starting exactly 2 full_finetune runs ==========")
    for i, combo in enumerate(FULL_COMBOS, start=1):
        run_name = f"3{i:02d}_full_finetune_lr_{sanitize(combo.lr)}_wd_{sanitize(combo.weight_decay)}"
        append_manifest(
            manifest,
            run_name,
            "full_finetune",
            "all",
            "full",
            "full",
            combo.lr,
            combo.weight_decay,
            args.sweep_root / run_name,
        )
        run_index += 1
        ok = run_exp(
            args=args,
            run_index=run_index,
            expected_runs=total_expected,
            run_name=run_name,
            metrics_rel=Path(args.dataset) / "full_finetune" / "metrics.json",
            train_args=[
                "--stage", "full_finetune",
                "--epochs", str(args.full_epochs),
                "--lr", combo.lr,
                "--weight-decay", combo.weight_decay,
            ],
        )
        fail_count += 0 if ok else 1

    if args.dry_run:
        print("Dry run complete; no training was launched.")
    else:
        print("========== Building summary ==========")
        build_summary(args.sweep_root, summary_csv)

    print("========== Sweep complete ==========")
    print(f"Sweep root: {args.sweep_root}")
    print(f"Manifest: {manifest}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Runs attempted: {run_index}")
    print(f"Failures: {fail_count}")
    print("Expected count: 19 total = 1 linear + 8 quad + 8 lora + 2 full")

    if fail_count:
        print(f"WARNING: {fail_count} run(s) failed. Check FAILED files and per-run run.log files.")


if __name__ == "__main__":
    main()
