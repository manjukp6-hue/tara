#!/usr/bin/env bash
# ==============================================================================
# TARA Native Google TPU Autonomous Multi-Generation Launcher Script
# ==============================================================================
# Supports:
#   - Google Cloud TPU v6e (Trillium | 1 / 4 / 8 chips | 32GB HBM)
#   - Google Cloud TPU v5p (Flagship LLM Training | 95GB HBM)
#   - Google Cloud TPU v5e (v5e-1 Colab / v5e-8 Kaggle / GCP LitePod)
#
# Usage:
#   bash run_tpu.sh                   # Standard multi-core TPU pre-training
#   bash run_tpu.sh --resume          # Resume from last checkpoint
#   bash run_tpu.sh --stage-id stage1 # Run specific stage on TPU
#
# Dedicated TPU runner: 100% isolated from GPU training (run_train.sh).
# Writes checkpoints safely to storage/models/checkpoints_tpu/
# ==============================================================================
set -euo pipefail

echo "=============================================================================="
echo "  TARA CLOUD TPU TRAINING PIPELINE (ALL TPU GENERATIONS RUNNER)"
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

# 2. Configure Google TPU Hardware Environment Flags (PJRT + BF16)
export PJRT_DEVICE="${PJRT_DEVICE:-TPU}"
export XLA_USE_BF16="${XLA_USE_BF16:-1}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"

# 3. Detect TPU Environment (Colab vs Kaggle vs Cloud TPU VM)
echo "[TPU SETUP] Inspecting Cloud TPU Environment..."
EXTRA_ARGS=()

TPU_TYPE_LOWER="$(echo "${TPU_ACCELERATOR_TYPE:-${TPU_NAME:-}}" | tr '[:upper:]' '[:lower:]')"

if [ -n "${KAGGLE_KERNEL_RUN_TYPE:-}" ] || [ -n "${KAGGLE_TPU_VERSION:-}" ]; then
    echo "[TPU DETECTED] Kaggle Cloud TPU Environment detected: TPU v5e-8 (8 Cores | 128GB HBM)"
    export TPU_NUM_DEVICES="${TPU_NUM_DEVICES:-8}"

elif [[ "$TPU_TYPE_LOWER" == *"v6e-1"* ]] || [[ "$TPU_TYPE_LOWER" == *"v5e-1"* ]] || [ -n "${COLAB_TPU_ADDR:-}" ] || [ "${TPU_NUM_DEVICES:-}" = "1" ]; then
    echo "[TPU DETECTED] Single-Core TPU Environment detected (e.g. Google Colab TPU v5e-1 / v6e-1)"
    export TPU_NUM_DEVICES=1
    EXTRA_ARGS+=("--single-core")

elif [[ "$TPU_TYPE_LOWER" == *"v6e-4"* ]]; then
    echo "[TPU DETECTED] Google Cloud TPU v6e-4 (Trillium 4 Cores | 128GB HBM)"
    export TPU_NUM_DEVICES="${TPU_NUM_DEVICES:-4}"

elif [[ "$TPU_TYPE_LOWER" == *"v6e"* ]] || [[ "$TPU_TYPE_LOWER" == *"trillium"* ]]; then
    echo "[TPU DETECTED] Google Cloud TPU v6e-8 (Trillium 8 Cores | 256GB HBM | 6th Gen)"
    export TPU_NUM_DEVICES="${TPU_NUM_DEVICES:-8}"

elif [[ "$TPU_TYPE_LOWER" == *"v5p"* ]]; then
    echo "[TPU DETECTED] Google Cloud TPU v5p (Flagship LLM Accelerator | 95GB HBM/chip)"
    export TPU_NUM_DEVICES="${TPU_NUM_DEVICES:-8}"

elif [ -n "${TPU_ACCELERATOR_TYPE:-}" ] || [ -n "${TPU_NAME:-}" ]; then
    echo "[TPU DETECTED] Cloud TPU Accelerator detected: ${TPU_ACCELERATOR_TYPE:-${TPU_NAME:-TPU}}"
    export TPU_NUM_DEVICES="${TPU_NUM_DEVICES:-8}"

else
    echo "[TPU NOTICE] Standard TPU environment flags configured (PJRT_DEVICE=$PJRT_DEVICE, BF16=$XLA_USE_BF16)."
    export TPU_NUM_DEVICES="${TPU_NUM_DEVICES:-8}"
fi

# 4. Optional Cloud Storage & Backup Sync Setup (Google Drive / Hugging Face)
if [ -n "${COLAB_TPU_ADDR:-}" ] || [ -d "/content" ]; then
    echo "[GDRIVE] Detecting Google Drive environment..."
    $PY_CMD -c "from gdrive_sync import mount_google_drive; mount_google_drive()" 2>/dev/null || true
fi

# Authoritative Default Dataset and Model Repositories (Exact Parity with run_train.sh)
export HF_DATASET_REPO="${HF_DATASET_REPO:-${HF_REPO:-manjukp6/tara_dataset_v1}}"
export HF_MODEL_REPO="${HF_MODEL_REPO:-${HF_REPO:-manjukp6/tara}}"
EXTRA_ARGS+=("--hf-dataset-repo" "$HF_DATASET_REPO")

HF_REPO_VAL="${HF_MODEL_REPO:-${HF_REPO:-}}"
if [ -n "$HF_REPO_VAL" ]; then
    EXTRA_ARGS+=("--hf-repo" "$HF_REPO_VAL")
fi

# 4a. Memory-Safe TPU Hyperparameter Defaults (Prevents 16GB HBM exhaustion)
EXTRA_ARGS+=("--batch-size" "${BATCH_SIZE:-1}")
EXTRA_ARGS+=("--grad-accum" "${GRAD_ACCUM:-32}")
EXTRA_ARGS+=("--loss-chunk-size" "${LOSS_CHUNK_SIZE:-512}")
EXTRA_ARGS+=("--precision" "${PRECISION:-bf16}")
if [ "${OFFLOAD_OPTIMIZER:-false}" = "true" ] || [ "${OFFLOAD_OPTIMIZER:-0}" = "1" ]; then
    EXTRA_ARGS+=("--offload-optimizer")
fi
LR="${LR:-3e-4}"
EXTRA_ARGS+=("--lr" "$LR")

# 4b. Production Ephemeral Key Safeguard & Pretraining Public Key Resolution
if [ "${TARA_ALLOW_EPHEMERAL_KEY:-0}" = "1" ] || [ "${TARA_ALLOW_EPHEMERAL_KEY:-0}" = "true" ]; then
    echo "[SECURITY WARNING] TARA_ALLOW_EPHEMERAL_KEY was enabled! Disabling strictly for production pretraining..."
    unset TARA_ALLOW_EPHEMERAL_KEY
fi
export TARA_ALLOW_EPHEMERAL_KEY="0"

# 4c. Authoritative Multi-Domain Curriculum Weights (Exact 1:1 Parity with run_train.sh)
DOMAIN_WEIGHTS="${DOMAIN_WEIGHTS:-02_code:0.25,01_math:0.15,04_kannada:0.15,12_reasoning:0.15,03_stem:0.10,08_world_model:0.05,11_tool_use:0.05,13_debugging:0.02,sft:0.02,05_philosophy:0.012,06_social:0.012,07_engineering:0.012,09_audio:0.012,10_vision:0.012}"
export DOMAIN_WEIGHTS
EXTRA_ARGS+=("--domain-weights" "$DOMAIN_WEIGHTS")

if [ -z "${TARA_AUTH_PUBLIC_KEY:-}" ]; then
    RESOLVED_PUBKEY=$($PY_CMD -c "
import os, re, json
try:
    from google.colab import userdata
    k = userdata.get('TARA_AUTH_PUBLIC_KEY')
    if k and len(k.strip()) == 64:
        print(k.strip())
        exit(0)
except Exception: pass
for f in ('model_config/tara_auth_pubkey.hex', 'storage/vault/tara_auth_pubkey.hex'):
    if os.path.isfile(f):
        c = open(f).read().strip()
        if len(c) == 64 and re.fullmatch(r'[0-9a-fA-F]{64}', c):
            print(c)
            exit(0)
" 2>/dev/null || true)
    if [ -n "$RESOLVED_PUBKEY" ]; then
        export TARA_AUTH_PUBLIC_KEY="$RESOLVED_PUBKEY"
        echo "[SECURITY] Authoritative Root Ed25519 authorization public key resolved."
    fi
fi

# 5. Determine Execution Mode (Single-Stage vs Full Multi-Stage Pipeline)
RUN_ALL_STAGES=0
FORWARD_ARGS=()

for arg in "$@"; do
    if [ "$arg" = "--all-stages" ] || [ "$arg" = "all" ]; then
        RUN_ALL_STAGES=1
    else
        FORWARD_ARGS+=("$arg")
    fi
done

TPU_CP_BASE="${TPU_CP_BASE:-storage/models/checkpoints_tpu}"
mkdir -p "$TPU_CP_BASE"

if [ "$RUN_ALL_STAGES" -eq 1 ]; then
    echo "=============================================================================="
    echo "  LAUNCHING AUTONOMOUS MULTI-STAGE TPU PIPELINE (STAGES 0..5)"
    echo "=============================================================================="
    STAGES=("stage0_pretraining" "stage1_unified" "stage2_abilities" "stage3_continual" "stage4_scale" "stage5_frontier")

    for i in "${!STAGES[@]}"; do
        STAGE_ID="${STAGES[$i]}"
        STAGE_OUTPUT_DIR="$TPU_CP_BASE/$STAGE_ID"
        STAGE_PROMOTED_DIR="storage/models/promoted_tpu_$STAGE_ID"
        STAGE_MODEL_DIR="storage/models/tara_candidate_v1"

        # Stage handoff: validate and seed from prior stage output
        if [ "$i" -gt 0 ]; then
            PREV_STAGE_ID="${STAGES[$((i-1))]}"
            PREV_OUTPUT_DIR="$TPU_CP_BASE/$PREV_STAGE_ID"
            PREV_PROMOTED_DIR="storage/models/promoted_tpu_$PREV_STAGE_ID"

            # Checkpoint Validation Gate: Deep validation of keys, shapes, NaNs, config, tokenizer
            VALID_HANDOFF_DIR=""
            for CANDIDATE in "$PREV_OUTPUT_DIR" "$PREV_PROMOTED_DIR"; do
                if [ -d "$CANDIDATE" ]; then
                    if $PY_CMD train_tpu.py --validate-checkpoint-handoff "$CANDIDATE" --stage-id "$PREV_STAGE_ID"; then
                        VALID_HANDOFF_DIR="$CANDIDATE"
                        break
                    fi
                fi
            done

            if [ -z "$VALID_HANDOFF_DIR" ]; then
                echo "[FATAL ERROR] Stage $STAGE_ID handoff validation failed!" >&2
                echo "  Neither $PREV_OUTPUT_DIR nor $PREV_PROMOTED_DIR passed deep checkpoint integrity verification." >&2
                echo "  Cannot promote unverified or corrupted model to $STAGE_ID. Stopping multi-stage pipeline immediately." >&2
                exit 1
            fi
            STAGE_MODEL_DIR="$VALID_HANDOFF_DIR"
            echo "[STAGE HANDOFF] Deep-validated $PREV_STAGE_ID checkpoint; seeding $STAGE_ID candidate weights from: $STAGE_MODEL_DIR"
        fi

        echo "------------------------------------------------------------------------------"
        echo "[PIPELINE EXECUTION] Starting Stage $((i)): $STAGE_ID on TPU"
        echo "  Model Input Dir : $STAGE_MODEL_DIR"
        echo "  Output Dir      : $STAGE_OUTPUT_DIR"
        echo "  Promoted Dir    : $STAGE_PROMOTED_DIR"
        echo "------------------------------------------------------------------------------"

        $PY_CMD train_tpu.py \
            --stage-id "$STAGE_ID" \
            --model-dir "$STAGE_MODEL_DIR" \
            --output-dir "$STAGE_OUTPUT_DIR" \
            --promoted-dir "$STAGE_PROMOTED_DIR" \
            "${EXTRA_ARGS[@]}" \
            "${FORWARD_ARGS[@]}"

        STAGE_STATUS=$?
        if [ $STAGE_STATUS -ne 0 ]; then
            echo "[FATAL ERROR] Stage $STAGE_ID exited with error code $STAGE_STATUS. Stopping multi-stage pipeline immediately!" >&2
            exit $STAGE_STATUS
        fi
        echo "[STAGE COMPLETE] Stage $STAGE_ID finished successfully on TPU."
    done

    echo "=============================================================================="
    echo "  ALL TPU STAGES (0..5) COMPLETED SUCCESSFULLY!"
    echo "=============================================================================="
    exit 0
fi

# 6. Single-Stage Execution (Preserves exact backwards compatibility)
TPU_OUTPUT_DIR="${TPU_OUTPUT_DIR:-$TPU_CP_BASE/stage0_pretraining}"
mkdir -p "$TPU_OUTPUT_DIR"
mkdir -p "storage/models/promoted_tpu_stage0"

echo "[STORAGE] Dedicated TPU Checkpoint Dir: $TPU_OUTPUT_DIR"
echo "[ISOLATION] GPU Checkpoints (storage/models/checkpoints/) remain 100% UNTOUCHED."
echo "------------------------------------------------------------------------------"

echo "[LAUNCH] Launching TARA Native TPU Trainer (Single Stage)..."
$PY_CMD train_tpu.py \
    --output-dir "$TPU_OUTPUT_DIR" \
    --resume \
    "${EXTRA_ARGS[@]}" \
    "${FORWARD_ARGS[@]}"

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
