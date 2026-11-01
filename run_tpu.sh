#!/usr/bin/env bash
# ==============================================================================
# TARA Native Google TPU v3-8 Autonomous Launcher Script
# ==============================================================================
# Usage:
#   bash run_tpu.sh                   # Standard 8-core TPU v3-8 pre-training
#   bash run_tpu.sh --resume          # Resume from last TPU checkpoint
#   bash run_tpu.sh --stage-id stage1 # Run specific stage on TPU
#
# Dedicated TPU runner: 100% isolated from GPU training (run_train.sh).
# Writes checkpoints safely to storage/models/checkpoints_tpu/
# ==============================================================================
set -euo pipefail

echo "=============================================================================="
echo "  TARA CLOUD TPU TRAINING PIPELINE (GOOGLE TPU v3-8 RUNNER)"
echo "=============================================================================="

# 1. Resolve Python executable
if command -v python3 &>/dev/null; then
    PY_CMD="python3"
elif command -v python &>/dev/null; then
    PY_CMD="python"
else
    echo "[ERROR] Python 3 not found in PATH." >&2
    exit 1
fi

# 2. Configure Google TPU Hardware Environment Flags (Supports v5e-8, v5e-1, v3-8, v2-8)
export PJRT_DEVICE="${PJRT_DEVICE:-TPU}"
export XLA_USE_BF16="${XLA_USE_BF16:-1}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"

# 3. Detect TPU Environment (Colab v5e-1 vs Kaggle v5e-8 vs Cloud TPU VM)
echo "[TPU SETUP] Inspecting Cloud TPU Environment..."
EXTRA_ARGS=()

if [ -n "${KAGGLE_KERNEL_RUN_TYPE:-}" ] || [ -n "${KAGGLE_TPU_VERSION:-}" ]; then
    echo "[TPU DETECTED] Kaggle Cloud TPU Environment detected: TPU v5e-8 (8 Cores | 128GB HBM)"
    export TPU_NUM_DEVICES="${TPU_NUM_DEVICES:-8}"
elif [ "${TPU_ACCELERATOR_TYPE:-}" = "v5e-1" ] || [ -n "${COLAB_TPU_ADDR:-}" ] && [ "${TPU_NUM_DEVICES:-}" = "1" ]; then
    echo "[TPU DETECTED] Google Colab TPU v5e-1 detected (1 Core | 16GB HBM)"
    export TPU_NUM_DEVICES=1
    EXTRA_ARGS+=("--single-core")
elif [ -n "${TPU_ACCELERATOR_TYPE:-}" ] || [ -n "${TPU_NAME:-}" ]; then
    echo "[TPU DETECTED] Cloud TPU Accelerator detected: ${TPU_ACCELERATOR_TYPE:-${TPU_NAME:-TPU v3-8/v5e}}"
    export TPU_NUM_DEVICES="${TPU_NUM_DEVICES:-8}"
else
    echo "[TPU NOTICE] Standard TPU environment flags configured (PJRT_DEVICE=$PJRT_DEVICE, BF16=$XLA_USE_BF16)."
    export TPU_NUM_DEVICES="${TPU_NUM_DEVICES:-8}"
fi

# 4. Prepare Dedicated TPU Directories (ISOLATED: Never overwrites GPU checkpoints)
TPU_OUTPUT_DIR="${TPU_OUTPUT_DIR:-storage/models/checkpoints_tpu/stage0_pretraining}"
mkdir -p "$TPU_OUTPUT_DIR"
mkdir -p "storage/models/promoted_tpu_stage0"

echo "[STORAGE] Dedicated TPU Checkpoint Dir: $TPU_OUTPUT_DIR"
echo "[ISOLATION] GPU Checkpoints (storage/models/checkpoints/) remain 100% UNTOUCHED."
echo "------------------------------------------------------------------------------"

# 5. Launch Native TPU Training (Multi-Core or Single-Core)
echo "[LAUNCH] Launching TARA Native TPU Trainer..."
$PY_CMD train_tpu.py \
    --output-dir "$TPU_OUTPUT_DIR" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
    "$@"

TPU_EXIT_CODE=$?
if [ "$TPU_EXIT_CODE" -eq 0 ]; then
    echo "=============================================================================="
    echo "  TARA TPU TRAINING COMPLETED SUCCESSFULLY!"
    echo "=============================================================================="
else
    echo "=============================================================================="
    echo "  [ERROR] TPU Training exited with code $TPU_EXIT_CODE"
    echo "=============================================================================="
    exit "$TPU_EXIT_CODE"
fi
