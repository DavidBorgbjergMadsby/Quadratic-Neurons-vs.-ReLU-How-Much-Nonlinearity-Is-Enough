#!/bin/bash
#BSUB -q gpuv100
#BSUB -J cifar10_relu_gate_sweep
#BSUB -n 8
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=32GB]"
#BSUB -W 48:00
#BSUB -o logs/cifar10_relu_gate_sweep_%J.out
#BSUB -e logs/cifar10_relu_gate_sweep_%J.err

set -euo pipefail

# ------------------------------------------------------------
# Cluster environment
# ------------------------------------------------------------
module purge
unset PYTHONHOME PYTHONPATH PYTHONSTARTUP PYTHONUSERBASE
module load python3/3.11.9
module load numpy/1.26.4-python-3.11.9-openblas-0.3.27

# Your files are in 3a_test, not 3_test.
cd ~/Desktop/Fagprojekt/3a_test || exit 1

# NOTE:
# LSF opens #BSUB -o/-e before this script runs.
# So run `mkdir -p logs` manually once before `bsub < this_file.sh`.
mkdir -p logs outputs data
mkdir -p "torch_cache_${LSB_JOBID:-manual}"

# Use a job-specific torch cache to avoid concurrent download/cache writes.
export TORCH_HOME="./torch_cache_${LSB_JOBID:-manual}"

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

# New Python file with --run-gate-sweep.
SCRIPT=${SCRIPT:-train_LoRA_Qudratic_relu_ablation_gate_sweep.py}

# Fallbacks for uploaded/downloaded copies.
if [ ! -f "${SCRIPT}" ] && [ -f "train_LoRA_Qudratic_relu_ablation_gate_sweep(1).py" ]; then
  SCRIPT="train_LoRA_Qudratic_relu_ablation_gate_sweep(1).py"
fi

if [ ! -f "${SCRIPT}" ] && [ -f "train_LoRA_Qudratic_relu_ablation_gate_sweep(2).py" ]; then
  SCRIPT="train_LoRA_Qudratic_relu_ablation_gate_sweep(2).py"
fi

if [ ! -f "${SCRIPT}" ]; then
  echo "ERROR: could not find training script: ${SCRIPT}"
  echo "Place train_LoRA_Qudratic_relu_ablation_gate_sweep.py in $(pwd), or set SCRIPT=/path/to/script.py"
  exit 1
fi

# Edit these arrays/values as needed.
SEEDS=(42 43 44 45 46)
GATES=(0.0 0.25 0.5 0.75 1.0)

LINEAR_EPOCHS=${LINEAR_EPOCHS:-10}
ADAPTER_EPOCHS=${ADAPTER_EPOCHS:-10}
BATCH_SIZE=${BATCH_SIZE:-32}
NUM_WORKERS=${NUM_WORKERS:-4}
QUAD_RANK=${QUAD_RANK:-4}
LR=${LR:-1e-3}
PRINT_NAMES=${PRINT_NAMES:-0}

# Unique per submitted LSF job.
SWEEP_ROOT=${SWEEP_ROOT:-./outputs/cifar10_relu_gate_sweep_${LSB_JOBID:-manual}}
mkdir -p "${SWEEP_ROOT}"

MANIFEST=${SWEEP_ROOT}/manifest.tsv
echo -e "seed\trelu_gates\tlinear_epochs\tadapter_epochs\tquad_rank\tlr\tsweep_root" > "${MANIFEST}"

print_names_arg=()
if [ "${PRINT_NAMES}" = "1" ]; then
  print_names_arg=(--print-trainable-names)
fi

echo "============================================================"
echo "SCRIPT: ${SCRIPT}"
echo "DATASET: ${DATASET}"
echo "SEEDS: ${SEEDS[*]}"
echo "GATES: ${GATES[*]}"
echo "SWEEP_ROOT: ${SWEEP_ROOT}"
echo "============================================================"

for SEED in "${SEEDS[@]}"; do
  echo "############################################################"
  echo "STARTING SEED ${SEED}"
  echo "For each gate, Python will train:"
  echo "  1) linear_base with that gate"
  echo "  2) quadratic_adapter using that gate-specific base"
  echo "  3) lora_adapter using that gate-specific base"
  echo "############################################################"

  cmd=(
    python -u "${SCRIPT}"
    --dataset "${DATASET}"
    --run-gate-sweep
    --relu-gates "${GATES[@]}"
    --data-root ./data
    --output-dir "${SWEEP_ROOT}"
    --linear-epochs "${LINEAR_EPOCHS}"
    --adapter-epochs "${ADAPTER_EPOCHS}"
    --batch-size "${BATCH_SIZE}"
    --num-workers "${NUM_WORKERS}"
    --lr "${LR}"
    --quad-rank "${QUAD_RANK}"
    --seed "${SEED}"
    "${print_names_arg[@]}"
  )

  echo "+ ${cmd[*]}"
  "${cmd[@]}"

  echo -e "${SEED}\t${GATES[*]}\t${LINEAR_EPOCHS}\t${ADAPTER_EPOCHS}\t${QUAD_RANK}\t${LR}\t${SWEEP_ROOT}" >> "${MANIFEST}"

  echo "############################################################"
  echo "FINISHED SEED ${SEED}"
  echo "############################################################"
done

echo "ALL EXPERIMENTS FINISHED"
echo "Sweep root: ${SWEEP_ROOT}"
echo "Manifest: ${MANIFEST}"

echo "Expected output layout example:"
echo "  ${SWEEP_ROOT}/${DATASET}/linear_base/relu_gate_0.5/seed_42/model.pt"
echo "  ${SWEEP_ROOT}/${DATASET}/quadratic_adapter/relu_gate_0.5/rank_${QUAD_RANK}/seed_42/model.pt"
echo "  ${SWEEP_ROOT}/${DATASET}/lora_adapter/relu_gate_0.5/rank_<auto>_alpha_<auto>/seed_42/model.pt"
