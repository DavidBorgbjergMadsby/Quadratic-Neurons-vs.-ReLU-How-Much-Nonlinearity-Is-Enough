#!/bin/bash
#BSUB -q gpuv100
#BSUB -J quadratic_cifar10
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 02:00
#BSUB -o logs/quadratic_%J.out
#BSUB -e logs/quadratic_%J.err

set -euo pipefail

module purge
unset PYTHONHOME PYTHONPATH PYTHONSTARTUP PYTHONUSERBASE
module load python3/3.11.9
module load numpy/1.26.4-python-3.11.9-openblas-0.3.27

cd ~/Desktop/Fagprojekt/1_test/1_test || exit 1

mkdir -p logs outputs data torch_cache
export TORCH_HOME=./torch_cache
source /tmp/$USER/fagprojekt_env/.venv/bin/activate

echo "Running on host:"
hostname
echo "GPU info:"
nvidia-smi
echo "Python:"
which python
python --version
python -c "import torch, torchvision, numpy; print('torch', torch.__version__); print('torchvision', torchvision.__version__); print('numpy', numpy.__version__); print('cuda?', torch.cuda.is_available())"

DATASET=cifar10
BASE_CKPT=./outputs/${DATASET}/linear_base/model.pt
QUAD_RANK=4
EPOCHS=10
BATCH_SIZE=32
LR=1e-3
NUM_WORKERS=4

if [ ! -f "${BASE_CKPT}" ]; then
  echo "ERROR: Missing base checkpoint: ${BASE_CKPT}"
  echo "Run train_linear_base_cifar10.sh first."
  exit 1
fi

python -u train_LoRA_Qudratic_with_scratch_only.py \
  --dataset ${DATASET} \
  --stage quadratic_adapter \
  --base-checkpoint ${BASE_CKPT} \
  --quad-rank ${QUAD_RANK} \
  --output-dir ./outputs \
  --epochs ${EPOCHS} \
  --batch-size ${BATCH_SIZE} \
  --lr ${LR} \
  --num-workers ${NUM_WORKERS} \
  --print-trainable-names
