#!/bin/bash
#BSUB -q gpuv100
#BSUB -J cifar10_relu_ablation
#BSUB -n 8
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=32GB]"
#BSUB -W 48:00
#BSUB -o logs/cifar10_relu_ablation_%J.out
#BSUB -e logs/cifar10_relu_ablation_%J.err

set -euo pipefail

# ------------------------------------------------------------
# Cluster environment, copied from the working L40S .sh style
# ------------------------------------------------------------
module purge
unset PYTHONHOME PYTHONPATH PYTHONSTARTUP PYTHONUSERBASE
module load python3/3.11.9
module load numpy/1.26.4-python-3.11.9-openblas-0.3.27

cd ~/Desktop/Fagprojekt/3_test || exit 1

mkdir -p logs outputs data torch_cache
export TORCH_HOME=./torch_cache
source ~/torch3119_clean/bin/activate

# ------------------------------------------------------------
# Basic diagnostics
# ------------------------------------------------------------
echo "Running on host:"
hostname
echo "GPU info:"
nvidia-smi
echo "Python:"
which python
python --version
python -c "import torch, torchvision, numpy; print('torch', torch.__version__); print('torchvision', torchvision.__version__); print('numpy', numpy.__version__); print('cuda?', torch.cuda.is_available())"

# ------------------------------------------------------------
# Experiment config
# ------------------------------------------------------------
DATASET=${DATASET:-cifar10}
SCRIPT=${SCRIPT:-train_LoRA_Qudratic_relu_ablation.py}

# Fallback for downloaded/uploaded copies with '(1)' in the filename.
if [ ! -f "${SCRIPT}" ] && [ -f "train_LoRA_Qudratic_relu_ablation(1).py" ]; then
  SCRIPT="train_LoRA_Qudratic_relu_ablation(1).py"
fi

if [ ! -f "${SCRIPT}" ]; then
  echo "ERROR: could not find training script: ${SCRIPT}"
  echo "Place train_LoRA_Qudratic_relu_ablation.py in $(pwd), or set SCRIPT=/path/to/script.py"
  exit 1
fi

# Same remaining seeds/gates as your PowerShell ablation script.
SEEDS=(42 43 44 45 46)
GATES=(0.0 0.25 0.5 0.75 1.0)

LINEAR_EPOCHS=${LINEAR_EPOCHS:-10}
ADAPTER_EPOCHS=${ADAPTER_EPOCHS:-10}
BATCH_SIZE=${BATCH_SIZE:-32}
NUM_WORKERS=${NUM_WORKERS:-4}
QUAD_RANK=${QUAD_RANK:-4}
LR=${LR:-1e-3}
PRINT_NAMES=${PRINT_NAMES:-0}

SWEEP_ROOT=${SWEEP_ROOT:-./outputs/cifar10_relu_ablation_l40s_${LSB_JOBID:-manual}}
mkdir -p "${SWEEP_ROOT}"
MANIFEST=${SWEEP_ROOT}/manifest.tsv

echo -e "seed\tstage\trelu_gate\tquad_rank\tlr\tepochs\toutput_dir\tbase_checkpoint" > "${MANIFEST}"

print_names_arg=()
if [ "${PRINT_NAMES}" = "1" ]; then
  print_names_arg=(--print-trainable-names)
fi

run_train() {
  local seed="$1"
  local stage="$2"
  local relu_gate="$3"
  local epochs="$4"
  local base_ckpt="${5:-}"

  local output_dir="${SWEEP_ROOT}/${DATASET}/${stage}/seed_${seed}"
  if [ "${stage}" = "quadratic_adapter" ] || [ "${stage}" = "lora_adapter" ]; then
    output_dir="${SWEEP_ROOT}/${DATASET}/${stage}/relu_gate_${relu_gate}/seed_${seed}"
  fi

  echo "============================================================"
  echo "DATASET ${DATASET} | SEED ${seed} | STAGE ${stage} | RELU_GATE ${relu_gate}"
  echo "Output dir will be: ${output_dir}"
  if [ -n "${base_ckpt}" ]; then
    echo "Base checkpoint: ${base_ckpt}"
  fi
  echo "============================================================"

  cmd=(
    python -u "${SCRIPT}"
    --dataset "${DATASET}"
    --stage "${stage}"
    --data-root ./data
    --output-dir "${SWEEP_ROOT}"
    --epochs "${epochs}"
    --batch-size "${BATCH_SIZE}"
    --num-workers "${NUM_WORKERS}"
    --lr "${LR}"
    --seed "${seed}"
    --relu-gate "${relu_gate}"
    "${print_names_arg[@]}"
  )

  if [ "${stage}" = "quadratic_adapter" ]; then
    cmd+=(--base-checkpoint "${base_ckpt}" --quad-rank "${QUAD_RANK}")
  elif [ "${stage}" = "lora_adapter" ]; then
    # Omitting --lora-rank lets the Python script auto-pick a rank matched
    # to the quadratic adapter's parameter budget for QUAD_RANK.
    cmd+=(--base-checkpoint "${base_ckpt}" --quad-rank "${QUAD_RANK}")
  fi

  echo "+ ${cmd[*]}"
  "${cmd[@]}"

  echo -e "${seed}\t${stage}\t${relu_gate}\t${QUAD_RANK}\t${LR}\t${epochs}\t${output_dir}\t${base_ckpt}" >> "${MANIFEST}"
}

for SEED in "${SEEDS[@]}"; do
  echo "############################################################"
  echo "STARTING SEED ${SEED}"
  echo "############################################################"

  # 1) Linear base uses normal ReLU. The uploaded Python script saves it at:
  #    ${SWEEP_ROOT}/${DATASET}/linear_base/seed_${SEED}/model.pt
  run_train "${SEED}" linear_base 1.0 "${LINEAR_EPOCHS}"

  BASE_CKPT="${SWEEP_ROOT}/${DATASET}/linear_base/seed_${SEED}/model.pt"
  if [ ! -f "${BASE_CKPT}" ]; then
    echo "ERROR: expected linear base checkpoint not found at ${BASE_CKPT}"
    exit 1
  fi

  # 2) ReLU gate ablation for quadratic and LoRA adapters.
  for GATE in "${GATES[@]}"; do
    run_train "${SEED}" quadratic_adapter "${GATE}" "${ADAPTER_EPOCHS}" "${BASE_CKPT}"
    run_train "${SEED}" lora_adapter      "${GATE}" "${ADAPTER_EPOCHS}" "${BASE_CKPT}"
  done

  echo "############################################################"
  echo "FINISHED SEED ${SEED}"
  echo "############################################################"
done

echo "ALL EXPERIMENTS FINISHED"
echo "Sweep root: ${SWEEP_ROOT}"
echo "Manifest: ${MANIFEST}"
