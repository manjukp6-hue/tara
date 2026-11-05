#!/usr/bin/env bash
set -euo pipefail

# Standalone World Model GPU Training Launcher for Colab / Kaggle
DATA_PATH="${1:-storage/datasets/08_commonsense_world_model}"
OUTPUT_DIR="${2:-storage/models/tara_world_model}"
EPOCHS="${3:-10}"
BATCH_SIZE="${4:-64}"
LR="${5:-0.0003}"

echo "=========================================================="
echo "  TARA Standalone Neural World Model GPU Training"
echo "  Dataset    : ${DATA_PATH}"
echo "  Output Dir : ${OUTPUT_DIR}"
echo "  Epochs     : ${EPOCHS}"
echo "  Batch Size : ${BATCH_SIZE}"
echo "  LR         : ${LR}"
echo "=========================================================="

python tara_training/world_model/world_train_gpu.py \
    --data-path "${DATA_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --lambda-dyn 2.0 \
    --lambda-rew 1.0 \
    --lambda-fail 1.0 \
    --lambda-val 0.5

echo "World Model GPU Training Completed Successfully."
