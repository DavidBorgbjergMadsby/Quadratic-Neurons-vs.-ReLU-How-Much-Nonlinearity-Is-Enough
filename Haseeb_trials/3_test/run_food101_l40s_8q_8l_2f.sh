#!/bin/bash
#BSUB -q gpul40s
#BSUB -J food101_8q8l2f
#BSUB -n 8
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=64GB]"
#BSUB -W 72:00
#BSUB -o logs/food101_8q8l2f_%J.out
#BSUB -e logs/food101_8q8l2f_%J.err

set -euo pipefail

# ============================================================
# Food101 ConvNeXt-Tiny sweep for one L40S GPU / LSF-BSUB HPC
# Exact run order:
#   1 linear_base
#   8 quad_dw runs
#   8 lora_dw runs
#   2 full_finetune runs
#
# Every experiment is isolated in its own output folder. A patched copy
# of train_LoRA_Qudratic.py is created so each run saves best_model.pt
# based on best test_acc, while your original training file is unchanged.
# ============================================================

# ----------------------------
# Cluster / environment setup
# ----------------------------
module purge
unset PYTHONHOME PYTHONPATH PYTHONSTARTUP PYTHONUSERBASE
module load python3/3.11.9
module load numpy/1.26.4-python-3.11.9-openblas-0.3.27

# Override these at submit time if your HPC paths differ, e.g.:
#   PROJECT_DIR=/path/to/project VENV_PATH=/path/to/venv bsub < run_food101_l40s_8q_8l_2f.sh
PROJECT_DIR=${PROJECT_DIR:-"$HOME/Desktop/Fagprojekt/1_test/1_test"}
VENV_PATH=${VENV_PATH:-"/tmp/$USER/fagprojekt_env/.venv"}

cd "$PROJECT_DIR" || exit 1

mkdir -p logs outputs data torch_cache
export TORCH_HOME=${TORCH_HOME:-"$PROJECT_DIR/torch_cache"}
source "$VENV_PATH/bin/activate"

# Avoid CPU oversubscription.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export PYTHONUNBUFFERED=1

echo "========== HPC job info =========="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Project dir: $PROJECT_DIR"
echo "Job ID: ${LSB_JOBID:-manual}"
echo "GPU info:"
nvidia-smi || true
echo "Python: $(which python)"
python --version
python - <<'PY'
import torch, torchvision, numpy
print('torch', torch.__version__)
print('torchvision', torchvision.__version__)
print('numpy', numpy.__version__)
print('cuda?', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu', torch.cuda.get_device_name(0))
PY

# ----------------------------
# Global experiment settings
# ----------------------------
DATASET=${DATASET:-food101}
DATA_ROOT=${DATA_ROOT:-data}
SEED=${SEED:-42}

# L40S should usually handle 128 for ConvNeXt-Tiny + Food101.
# Set BATCH_SIZE=64 at submit time if you hit OOM.
BATCH_SIZE=${BATCH_SIZE:-128}
NUM_WORKERS=${NUM_WORKERS:-8}
GRAD_CLIP=${GRAD_CLIP:-5.0}

LINEAR_EPOCHS=${LINEAR_EPOCHS:-30}
ADAPTER_EPOCHS=${ADAPTER_EPOCHS:-12}
FULL_EPOCHS=${FULL_EPOCHS:-25}

LINEAR_LR=${LINEAR_LR:-1e-3}
LINEAR_WD=${LINEAR_WD:-0.05}

# 0 = stop on first failed run. 1 = keep going and mark failed run folders.
CONTINUE_ON_FAIL=${CONTINUE_ON_FAIL:-1}

RUN_STAMP=${RUN_STAMP:-"${LSB_JOBID:-manual_$(date +%Y%m%d_%H%M%S)}"}
SWEEP_ROOT=${SWEEP_ROOT:-"$PROJECT_DIR/outputs/${DATASET}_l40s_8q8l2f_${RUN_STAMP}"}
mkdir -p "$SWEEP_ROOT"

TRAIN_SRC=${TRAIN_SRC:-train_LoRA_Qudratic.py}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-train_LoRA_Qudratic_hpc_best.py}

# ----------------------------
# Patch training script to save best_model.pt per run
# ----------------------------
echo "Creating patched training script with best_model.pt saving: $TRAIN_SCRIPT"
python - "$TRAIN_SRC" "$TRAIN_SCRIPT" <<'PY'
from pathlib import Path
import sys
src_path = Path(sys.argv[1])
dst_path = Path(sys.argv[2])
s = src_path.read_text()

if "best_test_acc = -float(\"inf\")" not in s:
    s = s.replace(
        "    start_time = time.time()\n\n    for epoch in range(args.epochs):\n",
        "    start_time = time.time()\n"
        "    best_test_acc = -float(\"inf\")\n"
        "    best_epoch = -1\n\n"
        "    for epoch in range(args.epochs):\n",
    )

    s = s.replace(
        "        history[\"test_acc\"].append(test_acc)\n\n        print(\n",
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
        "        print(\n",
    )

    s = s.replace(
        "    history[\"training_time_sec\"] = elapsed\n",
        "    history[\"training_time_sec\"] = elapsed\n"
        "    history[\"best_epoch\"] = best_epoch\n"
        "    history[\"best_test_acc\"] = best_test_acc\n",
    )

dst_path.write_text(s)
print(f"Wrote {dst_path}")
PY

python -m py_compile "$TRAIN_SCRIPT"

# ----------------------------
# Helpers
# ----------------------------
RUN_COUNT=0
FAIL_COUNT=0
MANIFEST="$SWEEP_ROOT/manifest.tsv"
SUMMARY_CSV="$SWEEP_ROOT/summary.csv"
printf "run_name\tstage\tscope\trank\thead\tlr\tweight_decay\toutput_dir\n" > "$MANIFEST"

sanitize() {
  echo "$1" | sed 's/+//g; s/-/m/g; s/\./p/g; s/=//g; s/ /_/g'
}

append_manifest() {
  local run_name="$1"
  local stage="$2"
  local scope="$3"
  local rank="$4"
  local head="$5"
  local lr="$6"
  local wd="$7"
  local outdir="$8"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$run_name" "$stage" "$scope" "$rank" "$head" "$lr" "$wd" "$outdir" >> "$MANIFEST"
}

run_exp() {
  local run_name="$1"; shift
  local metrics_rel="$1"; shift
  local run_dir="$SWEEP_ROOT/$run_name"
  local metrics_path="$run_dir/$metrics_rel"
  local donefile="$run_dir/DONE"
  local failedfile="$run_dir/FAILED"

  RUN_COUNT=$((RUN_COUNT + 1))
  mkdir -p "$run_dir"

  if [[ -f "$donefile" && -f "$metrics_path" ]]; then
    echo "[$RUN_COUNT] SKIP existing run: $run_name"
    return 0
  fi

  rm -f "$failedfile"
  echo "============================================================"
  echo "[$RUN_COUNT] RUN: $run_name"
  echo "Output: $run_dir"
  echo "Expected metrics: $metrics_path"
  echo "Started: $(date)"
  echo "============================================================"

  {
    echo "cd $PROJECT_DIR"
    printf 'python -u %q' "$TRAIN_SCRIPT"
    printf ' %q' "$@"
    printf ' --dataset %q --data-root %q --output-dir %q --batch-size %q --num-workers %q --grad-clip %q --seed %q --print-trainable-names\n' \
      "$DATASET" "$DATA_ROOT" "$run_dir" "$BATCH_SIZE" "$NUM_WORKERS" "$GRAD_CLIP" "$SEED"
  } > "$run_dir/command.txt"

  set +e
  python -u "$TRAIN_SCRIPT" "$@" \
    --dataset "$DATASET" \
    --data-root "$DATA_ROOT" \
    --output-dir "$run_dir" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --grad-clip "$GRAD_CLIP" \
    --seed "$SEED" \
    --print-trainable-names 2>&1 | tee "$run_dir/run.log"
  status=${PIPESTATUS[0]}
  set -e

  if [[ "$status" -ne 0 ]]; then
    echo "ERROR: run failed with status $status: $run_name" | tee "$failedfile"
    FAIL_COUNT=$((FAIL_COUNT + 1))
    if [[ "$CONTINUE_ON_FAIL" == "1" ]]; then
      return 0
    else
      exit "$status"
    fi
  fi

  if [[ ! -f "$metrics_path" ]]; then
    echo "ERROR: metrics file missing after successful run: $metrics_path" | tee "$failedfile"
    FAIL_COUNT=$((FAIL_COUNT + 1))
    if [[ "$CONTINUE_ON_FAIL" == "1" ]]; then
      return 0
    else
      exit 1
    fi
  fi

  date > "$donefile"
  echo "Finished: $(date)"
}

# ----------------------------
# 1) Linear base
# ----------------------------
LINEAR_RUN="000_linear_base_lr_$(sanitize "$LINEAR_LR")_wd_$(sanitize "$LINEAR_WD")"
append_manifest "$LINEAR_RUN" "linear_base" "none" "none" "head_only" "$LINEAR_LR" "$LINEAR_WD" "$SWEEP_ROOT/$LINEAR_RUN"
run_exp "$LINEAR_RUN" "$DATASET/linear_base/metrics.json" \
  --stage linear_base \
  --epochs "$LINEAR_EPOCHS" \
  --lr "$LINEAR_LR" \
  --weight-decay "$LINEAR_WD"

BASE_DIR="$SWEEP_ROOT/$LINEAR_RUN/$DATASET/linear_base"
BASE_CKPT="$BASE_DIR/best_model.pt"
if [[ ! -f "$BASE_CKPT" ]]; then
  BASE_CKPT="$BASE_DIR/model.pt"
fi
if [[ ! -f "$BASE_CKPT" ]]; then
  echo "ERROR: expected linear base checkpoint not found in $BASE_DIR"
  exit 1
fi
echo "Using base checkpoint for adapters: $BASE_CKPT"

# ----------------------------
# 2) Exactly 8 quadratic depthwise adapter runs
# Format: scope rank head lr weight_decay
# head is no/yes, where yes adds --train-head-with-adapter.
# ----------------------------
declare -a QUAD_COMBOS=(
  "last_stage 1 no  1e-3 0.0"
  "last_stage 2 no  1e-3 0.0"
  "last_stage 4 no  1e-3 0.0"
  "last_stage 2 no  3e-4 0.01"
  "all        1 no  3e-4 0.0"
  "all        2 no  3e-4 0.0"
  "last_stage 2 yes 1e-4 0.0"
  "all        2 yes 1e-4 0.0"
)

echo "========== Starting exactly 8 quad_dw runs =========="
quad_i=0
for combo in "${QUAD_COMBOS[@]}"; do
  read -r scope rank head lr wd <<< "$combo"
  quad_i=$((quad_i + 1))
  run_name="1$(printf '%02d' "$quad_i")_quad_dw_${scope}_r${rank}_${head}head_lr_$(sanitize "$lr")_wd_$(sanitize "$wd")"
  append_manifest "$run_name" "quad_dw" "$scope" "$rank" "$head" "$lr" "$wd" "$SWEEP_ROOT/$run_name"

  extra_head_arg=()
  if [[ "$head" == "yes" ]]; then
    extra_head_arg=(--train-head-with-adapter)
  fi

  run_exp "$run_name" "$DATASET/quad_dw/${scope}_rank_${rank}/metrics.json" \
    --stage quad_dw \
    --base-checkpoint "$BASE_CKPT" \
    --adapter-scope "$scope" \
    --adapter-rank "$rank" \
    "${extra_head_arg[@]}" \
    --epochs "$ADAPTER_EPOCHS" \
    --lr "$lr" \
    --weight-decay "$wd"
done

# ----------------------------
# 3) Exactly 8 LoRA depthwise adapter runs
# Same grid as quadratic for a fair comparison.
# ----------------------------
declare -a LORA_COMBOS=(
  "last_stage 1 no  1e-3 0.0"
  "last_stage 2 no  1e-3 0.0"
  "last_stage 4 no  1e-3 0.0"
  "last_stage 2 no  3e-4 0.01"
  "all        1 no  3e-4 0.0"
  "all        2 no  3e-4 0.0"
  "last_stage 2 yes 1e-4 0.0"
  "all        2 yes 1e-4 0.0"
)

echo "========== Starting exactly 8 lora_dw runs =========="
lora_i=0
for combo in "${LORA_COMBOS[@]}"; do
  read -r scope rank head lr wd <<< "$combo"
  lora_i=$((lora_i + 1))
  run_name="2$(printf '%02d' "$lora_i")_lora_dw_${scope}_r${rank}_${head}head_lr_$(sanitize "$lr")_wd_$(sanitize "$wd")"
  append_manifest "$run_name" "lora_dw" "$scope" "$rank" "$head" "$lr" "$wd" "$SWEEP_ROOT/$run_name"

  extra_head_arg=()
  if [[ "$head" == "yes" ]]; then
    extra_head_arg=(--train-head-with-adapter)
  fi

  run_exp "$run_name" "$DATASET/lora_dw/${scope}_rank_${rank}/metrics.json" \
    --stage lora_dw \
    --base-checkpoint "$BASE_CKPT" \
    --adapter-scope "$scope" \
    --adapter-rank "$rank" \
    "${extra_head_arg[@]}" \
    --epochs "$ADAPTER_EPOCHS" \
    --lr "$lr" \
    --weight-decay "$wd"
done

# ----------------------------
# 4) Exactly 2 full fine-tune runs
# Format: lr weight_decay
# ----------------------------
declare -a FULL_COMBOS=(
  "1e-4 0.01"
  "3e-5 0.05"
)

echo "========== Starting exactly 2 full_finetune runs =========="
full_i=0
for combo in "${FULL_COMBOS[@]}"; do
  read -r lr wd <<< "$combo"
  full_i=$((full_i + 1))
  run_name="3$(printf '%02d' "$full_i")_full_finetune_lr_$(sanitize "$lr")_wd_$(sanitize "$wd")"
  append_manifest "$run_name" "full_finetune" "all" "full" "full" "$lr" "$wd" "$SWEEP_ROOT/$run_name"
  run_exp "$run_name" "$DATASET/full_finetune/metrics.json" \
    --stage full_finetune \
    --epochs "$FULL_EPOCHS" \
    --lr "$lr" \
    --weight-decay "$wd"
done

# ----------------------------
# Final summary CSV
# ----------------------------
echo "========== Building summary =========="
python - "$SWEEP_ROOT" "$SUMMARY_CSV" <<'PY'
from pathlib import Path
import csv, json, sys
root = Path(sys.argv[1])
out = Path(sys.argv[2])
rows = []
for metrics_path in root.rglob('metrics.json'):
    try:
        data = json.loads(metrics_path.read_text())
    except Exception as e:
        print(f"Could not read {metrics_path}: {e}")
        continue
    test_acc = data.get('test_acc', [])
    test_loss = data.get('test_loss', [])
    train_acc = data.get('train_acc', [])
    train_loss = data.get('train_loss', [])
    if test_acc:
        best_i = max(range(len(test_acc)), key=lambda i: test_acc[i])
        best_epoch = best_i + 1
        best_test_acc = test_acc[best_i]
        best_test_loss = test_loss[best_i] if best_i < len(test_loss) else ''
        train_acc_at_best = train_acc[best_i] if best_i < len(train_acc) else ''
        train_loss_at_best = train_loss[best_i] if best_i < len(train_loss) else ''
        last_test_acc = test_acc[-1]
    else:
        best_epoch = data.get('best_epoch', '')
        best_test_acc = data.get('best_test_acc', '')
        best_test_loss = ''
        train_acc_at_best = ''
        train_loss_at_best = ''
        last_test_acc = ''

    rel = metrics_path.relative_to(root)
    run_name = rel.parts[0] if len(rel.parts) else metrics_path.parent.name
    rows.append({
        'run_name': run_name,
        'stage': data.get('stage', ''),
        'best_epoch': best_epoch,
        'best_test_acc': best_test_acc,
        'best_test_loss': best_test_loss,
        'train_acc_at_best': train_acc_at_best,
        'train_loss_at_best': train_loss_at_best,
        'last_test_acc': last_test_acc,
        'trainable_params': data.get('trainable_params', ''),
        'total_params': data.get('total_params', ''),
        'training_time_sec': data.get('training_time_sec', ''),
        'metrics_path': str(rel),
    })

rows.sort(key=lambda r: (float(r['best_test_acc']) if r['best_test_acc'] != '' else -1), reverse=True)
out.parent.mkdir(parents=True, exist_ok=True)
fieldnames = [
    'run_name','stage','best_epoch','best_test_acc','best_test_loss','train_acc_at_best',
    'train_loss_at_best','last_test_acc','trainable_params','total_params','training_time_sec','metrics_path'
]
with out.open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote summary: {out}")
print("Top runs:")
for r in rows[:20]:
    print(f"{r['best_test_acc']}\tepoch {r['best_epoch']}\t{r['stage']}\t{r['run_name']}")
PY

echo "========== Sweep complete =========="
echo "Sweep root: $SWEEP_ROOT"
echo "Manifest: $MANIFEST"
echo "Summary CSV: $SUMMARY_CSV"
echo "Runs attempted: $RUN_COUNT"
echo "Failures: $FAIL_COUNT"
echo "Expected count: 19 total = 1 linear + 8 quad + 8 lora + 2 full"
echo "Finished: $(date)"

if [[ "$FAIL_COUNT" -ne 0 ]]; then
  echo "WARNING: $FAIL_COUNT runs failed. Check FAILED files and per-run run.log files."
fi
