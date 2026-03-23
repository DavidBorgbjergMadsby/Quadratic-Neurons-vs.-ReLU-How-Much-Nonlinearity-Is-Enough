#!/bin/bash
#BSUB -q gpuv100
#BSUB -J resnet_cifar10
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 02:00
#BSUB -o logs/train_%J.out
#BSUB -e logs/train_%J.err

set -e

module purge
unset PYTHONHOME PYTHONPATH PYTHONSTARTUP PYTHONUSERBASE
module load python3/3.11.9
module load numpy/1.26.4-python-3.11.9-openblas-0.3.27

cd ~/Desktop/Fagprojekt/1_test/1_test || exit 1

export TORCH_HOME=./torch_cache
source .venv/bin/activate

echo "Running on host:"
hostname
echo "GPU info:"
nvidia-smi
echo "Python:"
which python
python --version

python -c "import torch, torchvision, numpy; print('torch', torch.__version__); print('torchvision', torchvision.__version__); print('numpy', numpy.__version__); print('cuda?', torch.cuda.is_available())"

DATASET=cifar10
HEAD=linear

python -u train_frozenbase.py \
  --dataset $DATASET \
  --head $HEAD \
  --output-dir ./outputs/${DATASET}_${HEAD}_${LSB_JOBID} \
  --epochs 5 \
  --batch-size 32 \
  --lr 1e-3 \
  --num-workers 4