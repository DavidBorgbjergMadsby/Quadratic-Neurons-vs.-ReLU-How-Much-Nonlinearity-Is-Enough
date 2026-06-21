#!/bin/bash
#BSUB -q gpua100
#BSUB -J food101_base_compact
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -W 48:00
#BSUB -o logs/food101_base_compact_%J.out
#BSUB -e logs/food101_base_compact_%J.err

set -euo pipefail

module purge
unset PYTHONHOME PYTHONPATH PYTHONSTARTUP PYTHONUSERBASE
module load python3/3.11.9
module load numpy/1.26.4-python-3.11.9-openblas-0.3.27

# Results will be saved from this folder.
cd ~/Desktop/Fagprojekt/5a_test || exit 1

mkdir -p logs outputs_convnext_base torch_cache
export TORCH_HOME=./torch_cache
source ~/torch3119_clean/bin/activate

# Keep using the existing Food101 data folder.
DATA_ROOT=~/Desktop/Fagprojekt/4_test/data
SCRIPT_NAME=train_LoRA_Qudratic_convnext_base.py

echo "Running on host:"
hostname
echo "GPU info:"
nvidia-smi
echo "Python:"
which python
python --version
python -c "import torch, torchvision, numpy; print('torch', torch.__version__); print('torchvision', torchvision.__version__); print('numpy', numpy.__version__); print('cuda?', torch.cuda.is_available())"

if [ ! -f "${SCRIPT_NAME}" ]; then
  echo "ERROR: ${SCRIPT_NAME} not found in $(pwd)"
  exit 1
fi

DATASET=food101
SEEDS=(420)

LINEAR_EPOCHS=30
ADAPTER_EPOCHS=16
FULL_EPOCHS=25

# ConvNeXt-Base is much larger than Tiny. Start with 32 on A100.
# If full fine-tuning runs out of memory, reduce to 16.
BATCH_SIZE=32
NUM_WORKERS=4
GRAD_CLIP=5.0
PRINT_NAMES=0

LINEAR_LR=1e-3
LINEAR_WD=0.05

# Atrous quadratic adapter option: 3x3 with dilation 3 gives a 7x7 effective receptive field.
QUAD_ADAPTER_KERNEL=3
QUAD_ADAPTER_DILATION=3

SWEEP_ROOT=./outputs_convnext_base/food101_a100_convnext_base_compact_${LSB_JOBID:-manual}
mkdir -p "${SWEEP_ROOT}"
MANIFEST=${SWEEP_ROOT}/manifest.tsv

echo -e "seed\trun_name\tstage\tscope\trank\ttrain_head\tlr\tweight_decay\toutput_dir" > "${MANIFEST}"

run_train() {
  local seed="$1"
  local run_name="$2"
  shift 2
  local output_dir="${SWEEP_ROOT}/seed_${seed}/${run_name}"
  mkdir -p "${output_dir}"

  echo "============================================================"
  echo "SEED ${seed} | RUN ${run_name}"
  echo "Output dir: ${output_dir}"
  echo "============================================================"

  python -u "${SCRIPT_NAME}" \
    --dataset "${DATASET}" \
    --data-root "${DATA_ROOT}" \
    --output-dir "${output_dir}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --grad-clip "${GRAD_CLIP}" \
    --seed "${seed}" \
    "$@" \
    $( [ "${PRINT_NAMES}" = "1" ] && echo --print-trainable-names )
}

run_adapter() {
  local seed="$1"
  local stage="$2"
  local idx="$3"
  local scope="$4"
  local rank="$5"
  local train_head="$6"
  local lr="$7"
  local wd="$8"
  local base_ckpt="$9"
  local quad_kernel="${10:-}"
  local quad_dilation="${11:-}"
  local adapter_alpha="${12:-}"

  local prefix="q"
  local stage_num="1"
  if [ "${stage}" = "lora_dw" ]; then
    prefix="l"
    stage_num="2"
  fi

  local scope_tag="ls"
  if [ "${scope}" = "all" ]; then
    scope_tag="all"
  fi

  local htag="h0"
  local head_label="no"
  local extra_head_arg=()
  if [ "${train_head}" = "1" ]; then
    htag="h1"
    head_label="yes"
    extra_head_arg=(--train-head-with-adapter)
  fi

  local safe_lr=${lr//-/m}
  safe_lr=${safe_lr//./p}
  local safe_wd=${wd//-/m}
  safe_wd=${safe_wd//./p}

  local quad_atrous_tag=""
  local extra_quad_atrous_arg=()
  if [ "${stage}" = "quad_dw" ] && [ -n "${quad_kernel}" ] && [ -n "${quad_dilation}" ]; then
    quad_atrous_tag="_k${quad_kernel}d${quad_dilation}"
    extra_quad_atrous_arg=(--quad-adapter-kernel-size "${quad_kernel}" --quad-adapter-dilation "${quad_dilation}")
  elif [ "${stage}" = "quad_dw" ] && { [ -n "${quad_kernel}" ] || [ -n "${quad_dilation}" ]; }; then
    echo "ERROR: quad adapter kernel and dilation must be provided together"
    exit 1
  fi

  local alpha_tag=""
  local extra_alpha_arg=()
  if [ -n "${adapter_alpha}" ]; then
    local safe_alpha=${adapter_alpha//-/m}
    safe_alpha=${safe_alpha//./p}
    alpha_tag="_a${safe_alpha}"
    extra_alpha_arg=(--adapter-alpha "${adapter_alpha}")
  fi

  local run_name
  printf -v run_name "%s%02d_%s_%s_r%s_%s_lr%s_wd%s%s%s" "${stage_num}" "${idx}" "${prefix}" "${scope_tag}" "${rank}" "${htag}" "${safe_lr}" "${safe_wd}" "${quad_atrous_tag}" "${alpha_tag}"

  echo -e "${seed}\t${run_name}\t${stage}\t${scope}\t${rank}\t${head_label}\t${lr}\t${wd}\t${SWEEP_ROOT}/seed_${seed}/${run_name}" >> "${MANIFEST}"

  run_train "${seed}" "${run_name}" \
    --stage "${stage}" \
    --base-checkpoint "${base_ckpt}" \
    --adapter-scope "${scope}" \
    --adapter-rank "${rank}" \
    --epochs "${ADAPTER_EPOCHS}" \
    --lr "${lr}" \
    --weight-decay "${wd}" \
    "${extra_head_arg[@]}" \
    "${extra_quad_atrous_arg[@]}" \
    "${extra_alpha_arg[@]}"
}

for SEED in "${SEEDS[@]}"; do
  echo "############################################################"
  echo "STARTING SEED ${SEED}"
  echo "############################################################"

  SEED_ROOT=${SWEEP_ROOT}/seed_${SEED}
  mkdir -p "${SEED_ROOT}"

  # 1) Linear base: required checkpoint for all adapter runs.
  LINEAR_RUN=000_lin_lr1e3_wd0p05
  echo -e "${SEED}\t${LINEAR_RUN}\tlinear_base\tnone\tnone\thead_only\t${LINEAR_LR}\t${LINEAR_WD}\t${SEED_ROOT}/${LINEAR_RUN}" >> "${MANIFEST}"

  run_train "${SEED}" "${LINEAR_RUN}" \
    --stage linear_base \
    --epochs "${LINEAR_EPOCHS}" \
    --lr "${LINEAR_LR}" \
    --weight-decay "${LINEAR_WD}"

  BASE_CKPT=${SEED_ROOT}/${LINEAR_RUN}/${DATASET}/linear_base/model.pt
  if [ ! -f "${BASE_CKPT}" ]; then
    echo "ERROR: expected linear base checkpoint not found at ${BASE_CKPT}"
    exit 1
  fi
  echo "Using base checkpoint for adapters: ${BASE_CKPT}"

  # 2) Three quadratic runs chosen from the best Tiny results:
  #    - best dense/non-atrous quadratic
  #    - best atrous quadratic
  #    - low-parameter atrous rank-1 alternative
  run_adapter "${SEED}" quad_dw 1 all 2 1 1e-4 0.0 "${BASE_CKPT}"
  run_adapter "${SEED}" quad_dw 2 all 2 1 3e-4 0.0 "${BASE_CKPT}" "${QUAD_ADAPTER_KERNEL}" "${QUAD_ADAPTER_DILATION}"
  run_adapter "${SEED}" quad_dw 3 all 1 1 1e-4 0.0 "${BASE_CKPT}" "${QUAD_ADAPTER_KERNEL}" "${QUAD_ADAPTER_DILATION}"

  # 3) Three LoRA runs near the best Tiny settings:
  #    - best LoRA setting
  #    - higher learning-rate variant
  #    - same best LR with weight decay regularization
  run_adapter "${SEED}" lora_dw 1 all 2 1 1e-4 0.0  "${BASE_CKPT}"
  run_adapter "${SEED}" lora_dw 2 all 2 1 3e-4 0.0  "${BASE_CKPT}"
  run_adapter "${SEED}" lora_dw 3 all 2 1 1e-4 0.05 "${BASE_CKPT}"

  # 4) One full fine-tuning run, selected from the best Tiny full-finetune result.
  FULL_RUN=301_f_lr1e4_wd0p01
  echo -e "${SEED}\t${FULL_RUN}\tfull_finetune\tall\tfull\tfull\t1e-4\t0.01\t${SEED_ROOT}/${FULL_RUN}" >> "${MANIFEST}"
  run_train "${SEED}" "${FULL_RUN}" \
    --stage full_finetune \
    --epochs "${FULL_EPOCHS}" \
    --lr 1e-4 \
    --weight-decay 0.01

  echo "############################################################"
  echo "FINISHED SEED ${SEED}"
  echo "############################################################"
done

echo "All seeds finished."
echo "Sweep root: ${SWEEP_ROOT}"
echo "Manifest: ${MANIFEST}"
