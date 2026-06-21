#!/bin/bash
#BSUB -q gpul40s
#BSUB -J food101_3scratch_seed
#BSUB -n 8
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=32GB]"
#BSUB -W 48:00
#BSUB -o logs/food101_3scratch_seed_%J.out
#BSUB -e logs/food101_3scratch_seed_%J.err

set -euo pipefail

module purge
unset PYTHONHOME PYTHONPATH PYTHONSTARTUP PYTHONUSERBASE
module load python3/3.11.9
module load numpy/1.26.4-python-3.11.9-openblas-0.3.27

# Go to the experiment folder where results should be saved.
cd ~/Desktop/Fagprojekt/5_test || exit 1

mkdir -p logs outputs torch_cache
export TORCH_HOME=./torch_cache
source ~/torch3119_clean/bin/activate

DATA_ROOT=~/Desktop/Fagprojekt/3a_test/data

# This script assumes you are using the Python file that contains these stages:
#   standard_relu_scratch
#   identity_scratch
#   quadratic_identity_scratch
# Change this if your updated training file has a different name.
TRAIN_SCRIPT=${TRAIN_SCRIPT:-train_LoRA_Qudratic.py}

if [ ! -f "${TRAIN_SCRIPT}" ]; then
  echo "ERROR: ${TRAIN_SCRIPT} not found in $(pwd)."
  echo "Either copy the updated training script here or run with:"
  echo "  TRAIN_SCRIPT=train_LoRA_Qudratic.py bsub < $(basename "$0")"
  exit 1
fi

echo "Running on host:"
hostname
echo "GPU info:"
nvidia-smi
echo "Python:"
which python
python --version
python -c "import torch, torchvision, numpy; print('torch', torch.__version__); print('torchvision', torchvision.__version__); print('numpy', numpy.__version__); print('cuda?', torch.cuda.is_available())"

DATASET=food101

# Edit this line for multi-seed experiments, e.g. SEEDS=(420 421 422 423 424).
SEEDS=(420)

# Same training budget for all three from-scratch comparisons.
SCRATCH_EPOCHS=100
BATCH_SIZE=128
NUM_WORKERS=4
GRAD_CLIP=5.0
PRINT_NAMES=1

# Shared optimizer settings.
SCRATCH_LR=1e-3
SCRATCH_WD=0.05

# Small controlled CNN width. Keep fixed across all three comparisons.
SCRATCH_WIDTH=64

# Quadratic rank for quadratic_identity_scratch.
QUAD_RANK=2

SWEEP_ROOT=./outputs/food101_l40s_3scratch_multiseed_${LSB_JOBID:-manual}
mkdir -p "${SWEEP_ROOT}"
MANIFEST=${SWEEP_ROOT}/manifest.tsv

echo -e "seed\trun_name\tstage\tactivation\tquadratic\tquad_rank\tscratch_width\tepochs\tlr\tweight_decay\toutput_dir" > "${MANIFEST}"

run_train() {
  local seed="$1"
  local run_name="$2"
  local stage="$3"
  local activation="$4"
  local quadratic="$5"
  local quad_rank="$6"

  local output_dir="${SWEEP_ROOT}/seed_${seed}/${run_name}"
  mkdir -p "${output_dir}"

  echo "============================================================"
  echo "SEED ${seed} | RUN ${run_name} | STAGE ${stage}"
  echo "Output dir: ${output_dir}"
  echo "============================================================"

  echo -e "${seed}\t${run_name}\t${stage}\t${activation}\t${quadratic}\t${quad_rank}\t${SCRATCH_WIDTH}\t${SCRATCH_EPOCHS}\t${SCRATCH_LR}\t${SCRATCH_WD}\t${output_dir}" >> "${MANIFEST}"

  python -u "${TRAIN_SCRIPT}" \
    --dataset "${DATASET}" \
    --stage "${stage}" \
    --data-root "${DATA_ROOT}" \
    --output-dir "${output_dir}" \
    --epochs "${SCRATCH_EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${SCRATCH_LR}" \
    --weight-decay "${SCRATCH_WD}" \
    --num-workers "${NUM_WORKERS}" \
    --grad-clip "${GRAD_CLIP}" \
    --seed "${seed}" \
    --scratch-width "${SCRATCH_WIDTH}" \
    --adapter-rank "${quad_rank}" \
    $( [ "${PRINT_NAMES}" = "1" ] && echo --print-trainable-names )
}

for SEED in "${SEEDS[@]}"; do
  echo "############################################################"
  echo "STARTING SEED ${SEED}"
  echo "############################################################"

  # 1) Standard pointwise nonlinearity baseline.
  # Random init, ReLU activations, train all parameters.
  run_train \
    "${SEED}" \
    "001_standard_relu_scratch" \
    "standard_relu_scratch" \
    "relu" \
    "no" \
    "${QUAD_RANK}"

  # 2) No-pointwise-nonlinearity control.
  # Random init, Identity activations, train all parameters.
  run_train \
    "${SEED}" \
    "002_identity_scratch" \
    "identity_scratch" \
    "identity" \
    "no" \
    "${QUAD_RANK}"

  # 3) Quadratic-neuron model without pointwise ReLU.
  # Random init, quadratic conv layers + Identity activations, train all parameters.
  run_train \
    "${SEED}" \
    "003_quadratic_identity_scratch_r${QUAD_RANK}" \
    "quadratic_identity_scratch" \
    "identity" \
    "yes" \
    "${QUAD_RANK}"

  echo "############################################################"
  echo "FINISHED SEED ${SEED}"
  echo "############################################################"
done

echo "All seeds finished."
echo "Sweep root: ${SWEEP_ROOT}"
echo "Manifest: ${MANIFEST}"
