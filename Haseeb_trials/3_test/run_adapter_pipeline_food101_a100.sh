#!/bin/bash
#BSUB -q gpua100
#BSUB -J adapters_food101
#BSUB -n 8
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=32GB]"
#BSUB -W 24:00
#BSUB -o logs/adapters_food101_pipeline_%J.out
#BSUB -e logs/adapters_food101_pipeline_%J.err

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

DATASET=food101
ADAPTER_RANK=2
ADAPTER_SCOPE=last_stage

LINEAR_EPOCHS=30
ADAPTER_EPOCHS=20
FULL_EPOCHS=25

# Good A100 default for ConvNeXt-Tiny + Food101
BATCH_SIZE=128
NUM_WORKERS=8

LINEAR_LR=1e-3
QUAD_LR=1e-2
LORA_LR=1e-3
FULL_LR=1e-4

WEIGHT_DECAY=0.05
GRAD_CLIP=5.0
OUTPUT_DIR=./outputs
BASE_CKPT=${OUTPUT_DIR}/${DATASET}/linear_base/model.pt

# 1) Linear probe: frozen ImageNet backbone + trainable classifier head.
python -u train_LoRA_Qudratic.py \
  --dataset ${DATASET} \
  --stage linear_base \
  --output-dir ${OUTPUT_DIR} \
  --epochs ${LINEAR_EPOCHS} \
  --batch-size ${BATCH_SIZE} \
  --lr ${LINEAR_LR} \
  --weight-decay ${WEIGHT_DECAY} \
  --grad-clip ${GRAD_CLIP} \
  --num-workers ${NUM_WORKERS} \
  --print-trainable-names

if [ ! -f "${BASE_CKPT}" ]; then
  echo "ERROR: expected linear base checkpoint not found at ${BASE_CKPT}"
  exit 1
fi

# 2) Quadratic adapter: rank 2, last stage, train classifier head with adapter.
python -u train_LoRA_Qudratic.py \
  --dataset ${DATASET} \
  --stage quad_dw \
  --base-checkpoint ${BASE_CKPT} \
  --adapter-scope ${ADAPTER_SCOPE} \
  --adapter-rank ${ADAPTER_RANK} \
  --train-head-with-adapter \
  --output-dir ${OUTPUT_DIR} \
  --epochs ${ADAPTER_EPOCHS} \
  --batch-size ${BATCH_SIZE} \
  --lr ${QUAD_LR} \
  --weight-decay ${WEIGHT_DECAY} \
  --grad-clip ${GRAD_CLIP} \
  --num-workers ${NUM_WORKERS} \
  --print-trainable-names

# 3) LoRA adapter: rank 2, last stage, same frozen linear-base checkpoint.
python -u train_LoRA_Qudratic.py \
  --dataset ${DATASET} \
  --stage lora_dw \
  --base-checkpoint ${BASE_CKPT} \
  --adapter-scope ${ADAPTER_SCOPE} \
  --adapter-rank ${ADAPTER_RANK} \
  --output-dir ${OUTPUT_DIR} \
  --epochs ${ADAPTER_EPOCHS} \
  --batch-size ${BATCH_SIZE} \
  --lr ${LORA_LR} \
  --weight-decay ${WEIGHT_DECAY} \
  --grad-clip ${GRAD_CLIP} \
  --num-workers ${NUM_WORKERS} \
  --print-trainable-names

# 4) Full fine-tuning: ImageNet-pretrained ConvNeXt-Tiny, all layers trainable.
python -u train_LoRA_Qudratic.py \
  --dataset ${DATASET} \
  --stage full_finetune \
  --output-dir ${OUTPUT_DIR} \
  --epochs ${FULL_EPOCHS} \
  --batch-size ${BATCH_SIZE} \
  --lr ${FULL_LR} \
  --weight-decay ${WEIGHT_DECAY} \
  --grad-clip ${GRAD_CLIP} \
  --num-workers ${NUM_WORKERS} \
  --print-trainable-names
