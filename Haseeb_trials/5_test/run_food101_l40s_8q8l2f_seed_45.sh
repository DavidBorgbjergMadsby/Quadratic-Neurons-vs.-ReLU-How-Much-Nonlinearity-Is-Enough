#!/bin/bash
#BSUB -q gpul40s
#BSUB -J food101_8q8l2f_seed
#BSUB -n 8
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=32GB]"
#BSUB -W 48:00
#BSUB -o logs/food101_8q8l2f_seed_%J.out
#BSUB -e logs/food101_8q8l2f_seed_%J.err

set -euo pipefail

module purge
unset PYTHONHOME PYTHONPATH PYTHONSTARTUP PYTHONUSERBASE
module load python3/3.11.9
module load numpy/1.26.4-python-3.11.9-openblas-0.3.27

# Go to the experiment folder where results should be saved
cd ~/Desktop/Fagprojekt/5_test || exit 1

mkdir -p logs outputs torch_cache
export TORCH_HOME=./torch_cache
source ~/torch3119_clean/bin/activate

DATA_ROOT=~/Desktop/Fagprojekt/4_test/data

echo "Running on host:"
hostname
echo "GPU info:"
nvidia-smi
echo "Python:"
which python
python --version
python -c "import torch, torchvision, numpy; print('torch', torch.__version__); print('torchvision', torchvision.__version__); print('numpy', numpy.__version__); print('cuda?', torch.cuda.is_available())"

DATASET=food101

# Same experiment grid as run_food101_local_8q_8l_2f_winfix.py, repeated for each seed.
# Edit this one line if you want more/fewer seeds.
SEEDS=(45)

LINEAR_EPOCHS=30
ADAPTER_EPOCHS=16
FULL_EPOCHS=25

BATCH_SIZE=128
NUM_WORKERS=4
GRAD_CLIP=5.0
PRINT_NAMES=1

LINEAR_LR=1e-3
LINEAR_WD=0.05

# QuadraNet V2-inspired atrous adapter option:
# ConvNeXt depthwise layers are 7x7. Using 3x3 with dilation 3 keeps
# a 7x7 effective receptive field while reducing Wa/Wb adapter compute.
QUAD_ADAPTER_KERNEL=3
QUAD_ADAPTER_DILATION=3

SWEEP_ROOT=./outputs/food101_l40s_8q8l2f_multiseed_${LSB_JOBID:-manual}
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

  python -u train_LoRA_Qudratic.py \
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

  local safe_lr=${lr//- /m}
  safe_lr=${safe_lr//- /m}
  safe_lr=${lr//-/m}
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

  # 1) Linear base.
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

  # 2) Focused quadratic runs based on previous Food101 results.
  # Previous runs showed adapter_scope=all, rank=2, train_head=yes, lr=1e-4, wd=0.0
  # was the strongest quadratic setting. Keep that as the anchor and test only nearby changes.
  # The optional quad kernel/dilation arguments enable the atrous adapter.
  # The final optional argument is --adapter-alpha. Omitting alpha preserves default quadratic scaling.
  run_adapter "${SEED}" quad_dw 9  all        2 1 1e-4 0.0  "${BASE_CKPT}" "${QUAD_ADAPTER_KERNEL}" "${QUAD_ADAPTER_DILATION}"
  run_adapter "${SEED}" quad_dw 10 all        2 1 3e-4 0.0  "${BASE_CKPT}" "${QUAD_ADAPTER_KERNEL}" "${QUAD_ADAPTER_DILATION}"
  run_adapter "${SEED}" quad_dw 11 all        2 1 3e-5 0.0  "${BASE_CKPT}" "${QUAD_ADAPTER_KERNEL}" "${QUAD_ADAPTER_DILATION}"
  run_adapter "${SEED}" quad_dw 12 all        1 1 1e-4 0.0  "${BASE_CKPT}" "${QUAD_ADAPTER_KERNEL}" "${QUAD_ADAPTER_DILATION}"
  run_adapter "${SEED}" quad_dw 13 all        2 1 1e-4 0.05 "${BASE_CKPT}" "${QUAD_ADAPTER_KERNEL}" "${QUAD_ADAPTER_DILATION}"
  run_adapter "${SEED}" quad_dw 14 all        2 1 1e-4 0.0  "${BASE_CKPT}" "${QUAD_ADAPTER_KERNEL}" "${QUAD_ADAPTER_DILATION}" 0.5
  run_adapter "${SEED}" quad_dw 15 all        2 1 1e-4 0.0  "${BASE_CKPT}" "${QUAD_ADAPTER_KERNEL}" "${QUAD_ADAPTER_DILATION}" 2.0

  # Lightweight ablation only: previous runs showed last_stage was clearly worse,
  # but this keeps one cheap comparison for the report.
  run_adapter "${SEED}" quad_dw 16 last_stage 2 1 1e-4 0.0  "${BASE_CKPT}" "${QUAD_ADAPTER_KERNEL}" "${QUAD_ADAPTER_DILATION}"

  # 3) Matched LoRA comparison near its previous best setting.
  run_adapter "${SEED}" lora_dw 9  all        2 1 1e-4 0.0  "${BASE_CKPT}"
  run_adapter "${SEED}" lora_dw 10 all        2 1 3e-4 0.0  "${BASE_CKPT}"
  run_adapter "${SEED}" lora_dw 11 all        1 1 1e-4 0.0  "${BASE_CKPT}"
  run_adapter "${SEED}" lora_dw 12 all        2 1 1e-4 0.05 "${BASE_CKPT}"

  # Lightweight ablation only.
  run_adapter "${SEED}" lora_dw 13 last_stage 2 1 1e-4 0.0  "${BASE_CKPT}"

  # 4) Exactly 2 full fine-tune runs, same combos as the Python file.
  FULL_RUN=301_f_lr1e4_wd0p01
  echo -e "${SEED}\t${FULL_RUN}\tfull_finetune\tall\tfull\tfull\t1e-4\t0.01\t${SEED_ROOT}/${FULL_RUN}" >> "${MANIFEST}"
  run_train "${SEED}" "${FULL_RUN}" \
    --stage full_finetune \
    --epochs "${FULL_EPOCHS}" \
    --lr 1e-4 \
    --weight-decay 0.01

  FULL_RUN=302_f_lr3e5_wd0p05
  echo -e "${SEED}\t${FULL_RUN}\tfull_finetune\tall\tfull\tfull\t3e-5\t0.05\t${SEED_ROOT}/${FULL_RUN}" >> "${MANIFEST}"
  run_train "${SEED}" "${FULL_RUN}" \
    --stage full_finetune \
    --epochs "${FULL_EPOCHS}" \
    --lr 3e-5 \
    --weight-decay 0.05

  echo "############################################################"
  echo "FINISHED SEED ${SEED}"
  echo "############################################################"
done

echo "All seeds finished."
echo "Sweep root: ${SWEEP_ROOT}"
echo "Manifest: ${MANIFEST}"
