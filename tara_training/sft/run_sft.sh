#!/usr/bin/env bash
# ==============================================================================
# TARA Autonomous SFT Runner — Colab & Kaggle Dual Support
# 100% Standalone Execution - Isolated from Pretraining Pipelines
# ==============================================================================

set -euo pipefail

echo "=============================================================================="
echo "  TARA AUTONOMOUS SFT RUNNER (COLAB & KAGGLE CLOUD ACCELERATION)"
echo "=============================================================================="

# 1. Environment Auto-Detection
IS_COLAB=0
IS_KAGGLE=0

if [ -d "/kaggle/working" ]; then
    IS_KAGGLE=1
    PLATFORM="Kaggle"
    WORK_DIR="/kaggle/working"
    OUTPUT_DIR="/kaggle/working/sft_checkpoints"
    DATA_DIR="/kaggle/working/datasets"
elif [ -d "/content" ]; then
    IS_COLAB=1
    PLATFORM="Google Colab"
    WORK_DIR="/content"
    OUTPUT_DIR="/content/sft_checkpoints"
    DATA_DIR="/content/datasets"
else
    PLATFORM="Standalone Linux Host"
    WORK_DIR="$(pwd)"
    OUTPUT_DIR="storage/models/sft_checkpoints"
    DATA_DIR="storage/datasets"
fi

echo "[PLATFORM] Detected: $PLATFORM"
echo "[WORKSPACE] Working directory: $WORK_DIR"
echo "[OUTPUT] SFT Checkpoints: $OUTPUT_DIR"

# 2. Hardware Inspection & VRAM Adaptation
GPU_MEM_MB=0
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "------------------------------------------------------------------------------"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true
    GPU_MEM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d ' ' || echo "0")
    echo "------------------------------------------------------------------------------"
fi

if [ -n "$GPU_MEM_MB" ] && [ "$GPU_MEM_MB" -ge 35000 ] 2>/dev/null; then
    # A100 / H100 (40GB / 80GB)
    BATCH_SIZE=1
    GRAD_ACCUM=8
    MAX_SEQ_LEN=2048
elif [ -n "$GPU_MEM_MB" ] && [ "$GPU_MEM_MB" -ge 14000 ] 2>/dev/null; then
    # Tesla T4 / P100 / L4 (15GB - 24GB, Standard Colab & Kaggle)
    BATCH_SIZE=1
    GRAD_ACCUM=16
    MAX_SEQ_LEN=1024
elif [ -n "$GPU_MEM_MB" ] && [ "$GPU_MEM_MB" -ge 7000 ] 2>/dev/null; then
    # 8GB VRAM
    BATCH_SIZE=1
    GRAD_ACCUM=16
    MAX_SEQ_LEN=512
else
    # Low-spec / CPU fallback
    BATCH_SIZE=1
    GRAD_ACCUM=32
    MAX_SEQ_LEN=256
fi

# Auto-install bitsandbytes for 8-bit AdamW on Colab / Kaggle
if [ "$IS_COLAB" -eq 1 ] || [ "$IS_KAGGLE" -eq 1 ]; then
    python3 -c "import bitsandbytes" 2>/dev/null || pip install -q bitsandbytes || true
fi

# 3. Environment Variables
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export HF_DATASET_REPO="${HF_DATASET_REPO:-manjukp6/tara_sft_v1}"
export HF_MODEL_REPO="${HF_MODEL_REPO:-manjukp6/tara}"

# 4. Dynamic Base Model Checkpoint Selection (Highest Step Auto-Discovery)
BASE_MODEL_DIR="${1:-${BASE_MODEL:-}}"

if [ -z "$BASE_MODEL_DIR" ] || [ ! -f "$BASE_MODEL_DIR/config.json" ]; then
    echo "[BASE MODEL] Dynamically resolving highest step base checkpoint across Local, GDrive, and Hugging Face..."
    RESOLVE_CMD="
import sys
sys.path.insert(0, '$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)')
from sft_train import resolve_highest_base_checkpoint
step, path = resolve_highest_base_checkpoint(
    hf_repo='$HF_MODEL_REPO',
    hf_token='${HF_TOKEN:-}',
    target_work_dir='$WORK_DIR'
)
print(path)
"
    RESOLVED_DIR=$(python3 -c "$RESOLVE_CMD" 2>/dev/null | tail -n1)
    if [ -n "$RESOLVED_DIR" ] && [ -f "$RESOLVED_DIR/config.json" ]; then
        BASE_MODEL_DIR="$RESOLVED_DIR"
    elif [ -d "storage/models/tara_candidate_v1" ]; then
        BASE_MODEL_DIR="storage/models/tara_candidate_v1"
    fi
fi

echo "[BASE MODEL] Dynamically selected base checkpoint: $BASE_MODEL_DIR"

# 5. SFT Dataset Synchronization from Hugging Face
if [ ! -d "$DATA_DIR" ] || [ -z "$(ls -A "$DATA_DIR" 2>/dev/null)" ]; then
    echo "[DATASET] Synchronizing SFT dataset from Hugging Face ($HF_DATASET_REPO)..."
    mkdir -p "$DATA_DIR"
    python3 -c "
from huggingface_hub import snapshot_download
try:
    snapshot_download(
        repo_id='$HF_DATASET_REPO',
        repo_type='dataset',
        local_dir='$DATA_DIR',
        max_workers=4
    )
    print('[DATASET] SFT dataset download complete!')
except Exception as e:
    print(f'[DATASET Warning] Failed to download dataset: {e}')
"
fi

# 6. Optional Pre-Tokenization Step (Binary Cache Compiler)
BIN_CACHE_DIR="${BIN_CACHE_DIR:-$WORK_DIR/sft_binary_cache}"
if [ "${PRETOKENIZE:-1}" = "1" ] && [ -d "$DATA_DIR" ] && [ -f "$BASE_MODEL_DIR/tokenizer.json" ]; then
    echo "[PRE-TOKENIZE] Compiling datasets into memory-mapped binary cache at $BIN_CACHE_DIR..."
    PRETOK_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/sft_pretokenize.py"
    python3 -u "$PRETOK_SCRIPT" \
        --data-dir "$DATA_DIR" \
        --tokenizer-path "$BASE_MODEL_DIR/tokenizer.json" \
        --output-dir "$BIN_CACHE_DIR" \
        --max-seq-len "$MAX_SEQ_LEN" || echo "[PRE-TOKENIZE Notice] Fallback to JIT multi-domain streaming if pre-tokenization skipped."
fi

# 7. Execute Isolated SFT Optimization Loop
SFT_STEPS="${MAX_STEPS:-1000}"
LR="${LR:-2e-5}"
DOMAIN_WEIGHTS="${DOMAIN_WEIGHTS:-04_kannada:0.30,02_code:0.25,12_reasoning:0.20,03_stem:0.15,sft:0.10}"

echo "=============================================================================="
echo "  STARTING SUPERVISED FINE-TUNING PIPELINE (SFT)"
echo "  Batch Size: $BATCH_SIZE | Grad Accum: $GRAD_ACCUM | Max Steps: $SFT_STEPS"
echo "  Sequence Length: $MAX_SEQ_LEN | Learning Rate: $LR"
echo "=============================================================================="

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/sft_train.py"

python3 -u "$SCRIPT_PATH" \
    --model-dir "$BASE_MODEL_DIR" \
    --data-dir "$DATA_DIR" \
    --bin-cache-dir "$BIN_CACHE_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --hf-repo "$HF_MODEL_REPO" \
    ${HF_TOKEN:+--hf-token "$HF_TOKEN"} \
    --batch-size "$BATCH_SIZE" \
    --grad-accum "$GRAD_ACCUM" \
    --lr "$LR" \
    --max-steps "$SFT_STEPS" \
    --max-seq-len "$MAX_SEQ_LEN" \
    --checkpoint-interval 100 \
    --probe-interval 50 \
    --benchmark-interval 50 \
    --domain-weights "$DOMAIN_WEIGHTS" \
    --gradient-checkpointing \
    --optimizer auto \
    --resume

echo "=============================================================================="
echo "  TARA SFT TRAINING PIPELINE COMPLETED SUCCESSFULLY!"
echo "=============================================================================="
