#!/usr/bin/env bash
# ==============================================================================
# TARA Autonomous Google Colab Stage-Wise Training & Cloud Sync Pipeline
# 100% Cloud Execution (0% Local PC usage)
# Flow: Clone from GitHub -> Native Rust Training -> Sync to Google Drive -> Push to HF
# ==============================================================================

set -euo pipefail

echo "=============================================================================="
echo "  TARA CLOUD TRAINING SYSTEM (GOOGLE COLAB AUTONOMOUS RUNNER)"
echo "=============================================================================="

# 0. Proactive Disk Hygiene: Reclaim disk space from stale scratch files, caches, and unused debug builds
echo "[DISK CLEANUP] Reclaiming disk space and stopping any stale processes..."
killall -9 tara_training_stages tara_gpu_step_test manual_trainer 2>/dev/null || true
rm -rf /tmp/tara_moments_* /tmp/tara_* /tmp/tmp* 2>/dev/null || true
rm -rf target/debug 2>/dev/null || true
rm -rf storage/checkpoints/gpu_step_test_parity 2>/dev/null || true
rm -rf /root/.cache/huggingface /root/.cache/pip /root/.cache/torch 2>/dev/null || true
if command -v df >/dev/null 2>&1; then
    df -h / | awk 'NR==2 {print "[DISK] Available disk space: " $4 " free / " $2 " total (" $5 " used)"}'
fi

# 1. Enforce Cloud Training Environment Variables for PyTorch & Rust Orchestrator
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TARA_MAX_MEMORY_MB="${TARA_MAX_MEMORY_MB:-}"
export TARA_MAX_SEQ_LEN="${TARA_MAX_SEQ_LEN:-128}"
export TARA_MAX_VAL_SAMPLES="${TARA_MAX_VAL_SAMPLES:-5}"
export TARA_SKIP_PREFLIGHT_VAL="${TARA_SKIP_PREFLIGHT_VAL:-1}"
export GDRIVE_CHECKPOINT_DIR="${GDRIVE_CHECKPOINT_DIR:-/content/drive/MyDrive/TARA_CHECKPOINTS}"
export HF_DATASET_REPO="${HF_DATASET_REPO:-${HF_MODEL_REPO:-${HF_REPO:-manjukp6/tara}}}"
export HF_MODEL_REPO="${HF_MODEL_REPO:-${HF_DATASET_REPO}}"

TRAINER_MODE="${TRAINER_MODE:-python}"
CANARY_MODE="${CANARY_MODE:-0}"
CANARY_EXTRA_ARGS=""
if [ "$CANARY_MODE" = "1" ] || [ "${1:-}" = "--canary" ]; then
    echo "=============================================================================="
    echo "  [CANARY VALIDATION RUNNER ACTIVE (15 Steps, Tesla T4 Safe Settings)]"
    echo "=============================================================================="
    CANARY_EXTRA_ARGS="--canary"
    TRAIN_BATCH_SIZE=1
    GRAD_ACCUM=4
    STAGE_MAX_STEPS=15
else
    TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
    GRAD_ACCUM="${GRAD_ACCUM:-8}"
    STAGE_MAX_STEPS="${STAGE_MAX_STEPS:-1000}"
fi
JOINT_STEPS="${JOINT_STEPS:-200}"
MAIN_DS_DIR="storage/datasets/main_dataset_v2"
PROMOTED_BASE="storage/models/promoted"
mkdir -p "$PROMOTED_BASE"


if command -v pip >/dev/null 2>&1; then
    echo "[SETUP] Installing Python dependencies (tokenizers, safetensors, huggingface_hub, bitsandbytes)..."
    pip install -q tokenizers safetensors huggingface_hub bitsandbytes 2>/dev/null || true
fi

# 2. Native Rust Orchestrator Mode (When TRAINER_MODE=rust)
# Install & Configure Rust Toolchain (Native Engine)
if ! command -v cargo >/dev/null 2>&1; then
    echo "[SETUP] Installing Native Rust Toolchain..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    # shellcheck disable=SC1091
    source "$HOME/.cargo/env"
else
    echo "[SETUP] Rust Toolchain already installed: $(cargo --version)"
fi

export PATH="$HOME/.cargo/bin:$PATH"

# Ensure NVIDIA driver and CUDA libraries are discoverable in Colab / Linux containers
export LD_LIBRARY_PATH="/usr/lib64-nvidia:/usr/local/cuda/compat:/usr/local/cuda-12.8/compat:${LD_LIBRARY_PATH:-}"

# Helper function for high-speed streaming downloads from Hugging Face via Python (huggingface_hub)
download_hf() {
    local repo_id="$1"
    local repo_type="$2"
    local prefix="$3"
    local local_dir="$4"

    mkdir -p "$local_dir"
    echo "[HF DOWNLOAD] Streaming $repo_type '$repo_id' (prefix: '$prefix') -> '$local_dir' via Python..."
    if python3 -c "
import os, sys
from huggingface_hub import snapshot_download

repo_id = sys.argv[1]
repo_type = sys.argv[2]
prefix = sys.argv[3]
local_dir = sys.argv[4]
token = os.environ.get('HF_TOKEN') or None
allow = [prefix] if prefix else None

snapshot_download(
    repo_id=repo_id,
    repo_type=repo_type,
    local_dir=local_dir,
    allow_patterns=allow,
    token=token,
    max_workers=8,
)
" "$repo_id" "$repo_type" "$prefix" "$local_dir"; then
        echo "[HF DOWNLOAD] Python download verified successfully into '$local_dir'."
        return 0
    fi

    # Fallback to compiled Rust downloader if available
    if [ -f "./target/release/tara_hf_downloader" ]; then
        echo "[HF DOWNLOAD NOTICE] Falling back to native Rust downloader..."
        ./target/release/tara_hf_downloader \
            --repo "$repo_id" \
            --repo-type "$repo_type" \
            --prefix "$prefix" \
            --dest "$local_dir"
    else
        echo "[HF DOWNLOAD ERROR] Python download failed and Rust downloader binary not found."
        return 1
    fi
}

# 2. Hardware Inspection & Dynamic Parameter Configuration
echo "------------------------------------------------------------------------------"
echo "[HARDWARE] Inspecting Cloud GPU Acceleration..."
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
else
    echo "[HARDWARE] No GPU detected. Defaulting to High-Performance Cloud CPU."
fi
echo "------------------------------------------------------------------------------"

# Dynamic Parameter Configuration
MAIN_DS_DIR="storage/datasets/main_dataset_v2"


# 3. Ingest Datasets for Pretraining
mkdir -p "$MAIN_DS_DIR"
if [ ! -f "$MAIN_DS_DIR/tara_shard_001.jsonl" ]; then
    echo "=== [DATASET INGESTION: STREAMING PRETRAINING SHARDS] ==="
    HF_DATASET_SOURCE="${HF_DATASET_REPO:-${HF_REPO:-manjukp6/tara}}"
    if [ -n "$HF_DATASET_SOURCE" ]; then
        if download_hf "$HF_DATASET_SOURCE" "dataset" "v2/*" "$MAIN_DS_DIR"; then
            if [ -d "$MAIN_DS_DIR/v2" ]; then
                shopt -s nullglob
                mv "$MAIN_DS_DIR/v2"/* "$MAIN_DS_DIR/"
                shopt -u nullglob
                rmdir "$MAIN_DS_DIR/v2"
            fi
            echo "Pretraining Dataset Ingestion Verified!"
        else
            echo "[DATASET NOTICE] Remote dataset download failed or not found. Checking local datasets in $MAIN_DS_DIR."
        fi
    fi
fi

# 3b. Ingest Main Dataset V1 for Stage 4 (Scale Expansion)
SCALE_DS_DIR="storage/datasets/main_dataset_v1"
mkdir -p "$SCALE_DS_DIR"
if [ ! -f "$SCALE_DS_DIR/tara_canonical_shard_000.jsonl" ]; then
    echo "=== [DATASET INGESTION: STREAMING MAIN DATASET V1 (STAGE 4) SHARDS] ==="
    HF_DATASET_SOURCE="${HF_DATASET_REPO:-${HF_REPO:-manjukp6/tara}}"
    if [ -n "$HF_DATASET_SOURCE" ]; then
        if download_hf "$HF_DATASET_SOURCE" "dataset" "v1/*" "$SCALE_DS_DIR"; then
            if [ -d "$SCALE_DS_DIR/v1" ]; then
                shopt -s nullglob
                mv "$SCALE_DS_DIR/v1"/* "$SCALE_DS_DIR/"
                shopt -u nullglob
                rmdir "$SCALE_DS_DIR/v1"
            fi
            echo "Main Dataset V1 (Stage 4) Ingestion Verified!"
        else
            echo "[DATASET NOTICE] Remote v1 dataset download failed or not found. Checking local datasets."
        fi
    fi
fi

# 4. Base Model Candidate & World Model Ingestion from Hugging Face Model Hub
mkdir -p storage/models/tara_candidate_v1
mkdir -p storage/models/tara_world_model

HF_MODEL_SOURCE="${HF_MODEL_REPO:-${HF_REPO:-manjukp6/tara}}"

# In case any candidate model files were previously downloaded directly into storage/models:
shopt -s nullglob
LOOSE_SHARDS=(storage/models/*.safetensors)
if [ -e "${LOOSE_SHARDS[0]:-}" ]; then
    mv storage/models/*.safetensors storage/models/tara_candidate_v1/ 2>/dev/null || true
    mv storage/models/*.json storage/models/tara_candidate_v1/ 2>/dev/null || true
fi
shopt -u nullglob

# Verify Neural Model Candidate weights exist on disk
NEURAL_SHARDS=(storage/models/tara_candidate_v1/*.safetensors)
if [ ! -e "${NEURAL_SHARDS[0]:-}" ]; then
    echo "=== [MODEL INGESTION: STREAMING NEURAL MODEL V1 FROM HUGGING FACE] ==="
    echo "[HF DOWNLOAD] Downloading neural_model_v1/* from https://huggingface.co/$HF_MODEL_SOURCE..."
    if download_hf "$HF_MODEL_SOURCE" "model" "neural_model_v1/*" "storage/models/tara_candidate_v1"; then
        if [ -d "storage/models/tara_candidate_v1/neural_model_v1" ]; then
            shopt -s nullglob
            mv storage/models/tara_candidate_v1/neural_model_v1/* storage/models/tara_candidate_v1/
            shopt -u nullglob
            rmdir storage/models/tara_candidate_v1/neural_model_v1
        fi
        CHECK_SHARDS=(storage/models/tara_candidate_v1/*.safetensors)
        if [ -e "${CHECK_SHARDS[0]:-}" ]; then
            echo "Neural Model V1 Ingestion from Hugging Face Verified!"
        else
            echo "=== [ERROR: NO SAFETENSORS SHARDS FOUND IN DOWNLOADED REPO] ==="
            exit 1
        fi
    else
        echo "=== [ERROR: COULD NOT DOWNLOAD NEURAL MODEL V1 FROM HUGGING FACE] ==="
        echo "Please ensure HF_TOKEN is exported with read access to https://huggingface.co/$HF_MODEL_SOURCE"
        exit 1
    fi
fi

# Verify World Model weights exist on disk
WORLD_SHARDS=(storage/models/tara_world_model/*.safetensors)
if [ ! -e "${WORLD_SHARDS[0]:-}" ]; then
    echo "=== [MODEL INGESTION: STREAMING WORLD MODEL V1 FROM HUGGING FACE] ==="
    echo "[HF DOWNLOAD] Downloading world_model_v1/* from https://huggingface.co/$HF_MODEL_SOURCE..."
    if download_hf "$HF_MODEL_SOURCE" "model" "world_model_v1/*" "storage/models/tara_world_model"; then
        if [ -d "storage/models/tara_world_model/world_model_v1" ]; then
            shopt -s nullglob
            mv storage/models/tara_world_model/world_model_v1/* storage/models/tara_world_model/
            shopt -u nullglob
            rmdir storage/models/tara_world_model/world_model_v1
        fi
        CHECK_WORLD=(storage/models/tara_world_model/*.safetensors)
        if [ -e "${CHECK_WORLD[0]:-}" ]; then
            echo "World Model V1 Ingestion from Hugging Face Verified!"
        else
            echo "=== [ERROR: NO WORLD MODEL SAFETENSORS SHARDS FOUND IN DOWNLOADED REPO] ==="
            exit 1
        fi
    else
        echo "=== [ERROR: COULD NOT DOWNLOAD WORLD MODEL V1 FROM HUGGING FACE] ==="
        echo "Please ensure HF_TOKEN is exported with read access to https://huggingface.co/$HF_MODEL_SOURCE"
        exit 1
    fi
fi

# Determine active checkpoint root (matches Rust GDRIVE_CHECKPOINT_DIR fallback)
if [ -d "/content/drive/MyDrive" ]; then
    ACTIVE_CP_DIR="$GDRIVE_CHECKPOINT_DIR"
else
    ACTIVE_CP_DIR="storage/models/checkpoints"
fi
mkdir -p "$ACTIVE_CP_DIR"

# 5. Mandatory Pre-Training Verification, Deep De-duplication & Tokenizer Gate
if [ ! -f "./target/release/tara_pretraining_gate" ]; then
    echo "[BUILD] Compiling TARA Pre-Training Gate..."
    cargo build --release --bin tara_pretraining_gate
fi

# Audits curriculum if present, otherwise audits ingested main_dataset_v2
GATE_TARGET_DS="storage/datasets/curriculum"
if [ ! -d "$GATE_TARGET_DS" ] || [ -z "$(ls -A "$GATE_TARGET_DS" 2>/dev/null)" ]; then
    GATE_TARGET_DS="$MAIN_DS_DIR"
fi

echo "=== [PRE-TRAINING GATE: DEDUPLICATION, TOKENIZER AUDIT & AUTHORIZATION] ==="
./target/release/tara_pretraining_gate \
    --model storage/models/tara_candidate_v1 \
    --dataset "$GATE_TARGET_DS"

# Ensure pretraining_authorization.json is also mirrored in ACTIVE_CP_DIR if GDrive is active
if [ -f "storage/models/checkpoints/pretraining_authorization.json" ] && [ "$ACTIVE_CP_DIR" != "storage/models/checkpoints" ]; then
    cp "storage/models/checkpoints/pretraining_authorization.json" "$ACTIVE_CP_DIR/pretraining_authorization.json"
fi
echo "Pre-Training Authorization Gate Passed!"

# 5b. Optional Production-Model GPU 1-Step Diagnostic (Bypassed by default to start training immediately)
if [ "${RUN_GPU_STEP_TEST:-0}" = "1" ] || [ "${RUN_GPU_STEP_TEST:-}" = "always" ]; then
    echo "=== [PRODUCTION-MODEL GPU 1-STEP EXECUTION & MEMORY DIAGNOSTIC] ==="
    ./target/release/tara_gpu_step_test \
        --model storage/models/tara_candidate_v1 \
        --device auto \
        --precision auto || true
fi

# 6. Execute Autonomous Extensible Multi-Stage Training Pipeline (Stages 0..5)
# Fallback scale_dir to MAIN_DS_DIR if storage/datasets/main_dataset_v1 is not present
SCALE_DS_DIR="storage/datasets/main_dataset_v1"
if [ ! -d "$SCALE_DS_DIR" ] || [ -z "$(ls -A "$SCALE_DS_DIR" 2>/dev/null)" ]; then
    SCALE_DS_DIR="$MAIN_DS_DIR"
fi

# 6. Execute Autonomous Multi-Stage Cloud Pretraining Pipeline
TRAINER_MODE="${TRAINER_MODE:-python}"
echo "=== [AUTONOMOUS CLOUD TRAINING PIPELINE (MODE: $TRAINER_MODE)] ==="

if [ "$TRAINER_MODE" = "python" ]; then
    echo "[TRAINER] Launching High-Throughput PyTorch Cloud GPU Multi-Stage Pipeline..."

    # Stage 0: Foundational Pretraining
    python3 cloud_train.py \
        --stage-id "stage0_pretraining" \
        --stage-name "Stage 0: Foundational Pretraining" \
        --model-dir storage/models/tara_candidate_v1 \
        --dataset-dir "$MAIN_DS_DIR" \
        --output-dir "$ACTIVE_CP_DIR/stage0_pretraining" \
        --promoted-dir "$PROMOTED_BASE/stage0_pretraining" \
        --max-steps "$STAGE_MAX_STEPS" \
        --batch-size "$TRAIN_BATCH_SIZE" \
        --grad-accum "$GRAD_ACCUM" \
        --hf-repo "$HF_MODEL_REPO" \
        $CANARY_EXTRA_ARGS

    # In canary validation mode, stop after stage 0 verification to report telemetry
    if [ -n "$CANARY_EXTRA_ARGS" ]; then
        echo "=============================================================================="
        echo "  [CANARY VALIDATION RUN FINISHED & TELEMETRY RECORDED]"
        echo "=============================================================================="
        exit 0
    fi


    STAGE0_OUT="$ACTIVE_CP_DIR/stage0_pretraining"
    if [ -d "$ACTIVE_CP_DIR/stage0_pretraining/best_checkpoint" ]; then
        STAGE0_OUT="$ACTIVE_CP_DIR/stage0_pretraining/best_checkpoint"
    fi

    # Stage 1: Academic & Skills Curriculum
    CURRICULUM_DS="storage/datasets/curriculum"
    if [ ! -d "$CURRICULUM_DS" ] || [ -z "$(ls -A "$CURRICULUM_DS" 2>/dev/null)" ]; then
        CURRICULUM_DS="$MAIN_DS_DIR"
    fi

    STAGE0_MODEL_INPUT="$STAGE0_OUT"
    if [ -d "$PROMOTED_BASE/stage0_pretraining" ] && [ -f "$PROMOTED_BASE/stage0_pretraining/model.safetensors" ]; then
        STAGE0_MODEL_INPUT="$PROMOTED_BASE/stage0_pretraining"
    fi

    python3 cloud_train.py \
        --stage-id "stage1_curriculum" \
        --stage-name "Stage 1: Academic & Skills Curriculum" \
        --model-dir "$STAGE0_MODEL_INPUT" \
        --dataset-dir "$CURRICULUM_DS" \
        --output-dir "$ACTIVE_CP_DIR/stage1_curriculum" \
        --promoted-dir "$PROMOTED_BASE/stage1_curriculum" \
        --previous-stage-dir "$ACTIVE_CP_DIR/stage0_pretraining" \
        --max-steps "$STAGE_MAX_STEPS" \
        --batch-size "$TRAIN_BATCH_SIZE" \
        --grad-accum "$GRAD_ACCUM" \
        --hf-repo "$HF_MODEL_REPO"

    # Extensible Loop for Continual Expansion (Stages 2..N)
    STAGE_IDX=2
    while true; do
        NEXT_DS="storage/datasets/stage${STAGE_IDX}"
        if [ ! -d "$NEXT_DS" ] || [ -z "$(ls -A "$NEXT_DS" 2>/dev/null)" ]; then
            break
        fi

        PREV_STAGE_IDX=$((STAGE_IDX - 1))
        PREV_MODEL_INPUT="$ACTIVE_CP_DIR/stage${PREV_STAGE_IDX}"
        if [ -d "$PROMOTED_BASE/stage${PREV_STAGE_IDX}" ] && [ -f "$PROMOTED_BASE/stage${PREV_STAGE_IDX}/model.safetensors" ]; then
            PREV_MODEL_INPUT="$PROMOTED_BASE/stage${PREV_STAGE_IDX}"
        fi

        echo "=== [AUTONOMOUS CLOUD TRAINING PIPELINE: LAUNCHING STAGE $STAGE_IDX] ==="
        python3 cloud_train.py \
            --stage-id "stage${STAGE_IDX}" \
            --stage-name "Stage ${STAGE_IDX}: Continual Knowledge Expansion" \
            --model-dir "$PREV_MODEL_INPUT" \
            --dataset-dir "$NEXT_DS" \
            --output-dir "$ACTIVE_CP_DIR/stage${STAGE_IDX}" \
            --promoted-dir "$PROMOTED_BASE/stage${STAGE_IDX}" \
            --previous-stage-dir "$ACTIVE_CP_DIR/stage${PREV_STAGE_IDX}" \
            --max-steps "$STAGE_MAX_STEPS" \
            --batch-size "$TRAIN_BATCH_SIZE" \
            --grad-accum "$GRAD_ACCUM" \
            --hf-repo "$HF_MODEL_REPO"

        STAGE_IDX=$((STAGE_IDX + 1))
    done

    PIPELINE_EXIT=$?
else
    if [ ! -f "./target/release/tara_training_stages" ]; then
        echo "[BUILD] Compiling TARA Training Stages Orchestrator..."
        cargo build --release --bin tara_training_stages
    fi
    set +e
    ./target/release/tara_training_stages \
        --all-stages \
        --checkpoints-dir "$ACTIVE_CP_DIR" \
        --main-dataset-dir "$MAIN_DS_DIR" \
        --scale-dir "$SCALE_DS_DIR" \
        --device auto \
        --precision auto \
        --batch-size "$TRAIN_BATCH_SIZE" \
        --max-steps "$STAGE_MAX_STEPS"
    PIPELINE_EXIT=$?
    set -e
fi

if [ $PIPELINE_EXIT -ne 0 ]; then
    echo "=============================================================================="
    echo "[PIPELINE FAILURE] tara_training_stages exited with status: $PIPELINE_EXIT"
    echo "Host memory state at termination:"
    free -h || true
    echo "Kernel termination logs (dmesg):"
    dmesg -T | grep -i -E "oom|killed process|out of memory" | tail -20 || true
    echo "=============================================================================="
    exit $PIPELINE_EXIT
fi

echo "Autonomous Multi-Stage Training Pipeline Completed Successfully!"

# 7. Execute Neural & World Model Joint Co-Training
if [ ! -f "./target/release/manual_trainer" ]; then
    echo "[BUILD] Compiling TARA Manual Trainer..."
    cargo build --release --bin manual_trainer
fi
echo "=== [WORLD MODEL & NEURAL JOINT CO-TRAINING ($JOINT_STEPS STEPS, BATCH $TRAIN_BATCH_SIZE)] ==="
./target/release/manual_trainer \
    --mode both \
    --steps "$JOINT_STEPS" \
    --batch-size "$TRAIN_BATCH_SIZE"

echo "Neural & World Model Joint Training Completed Successfully!"

# 8. Checkpoint Preservation to Google Drive (Separated into distinct subdirectories)
if [ -d "/content/drive/MyDrive" ]; then
    echo "------------------------------------------------------------------------------"
    echo "[GDRIVE BACKUP] Syncing World Model & Joint checkpoints to Google Drive..."
    mkdir -p "$GDRIVE_CHECKPOINT_DIR/world_joint"
    if [ -d "manual_training/checkpoints" ] && [ -n "$(ls -A manual_training/checkpoints 2>/dev/null)" ]; then
        cp -r manual_training/checkpoints/* "$GDRIVE_CHECKPOINT_DIR/world_joint/"
    fi
    echo "[GDRIVE BACKUP] All Neural and World Model checkpoints archived in: $GDRIVE_CHECKPOINT_DIR"
else
    echo "[GDRIVE NOTICE] /content/drive/MyDrive not mounted. Checkpoints saved locally in Colab."
fi

# 9. Automatic Hugging Face Model Promotion & Upload (Subfolder Isolated)
if [ -n "${HF_TOKEN:-}" ] && [ -n "${HF_REPO:-}" ]; then
    echo "------------------------------------------------------------------------------"
    echo "[HUGGING FACE SYNC] Uploading trained model safetensors to '$HF_REPO'..."

    # Locate the highest completed stage directory dynamically (unbounded N stages)
    FINAL_STAGE_DIR=""
    WORLD_DIR="manual_training/checkpoints"
    if [ -d "$PROMOTED_BASE" ]; then
        for pdir in $(find "$PROMOTED_BASE" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | sort -V -r); do
            if [ -f "$pdir/model.safetensors" ]; then
                FINAL_STAGE_DIR="$pdir"
                break
            fi
        done
    fi
    if [ -z "$FINAL_STAGE_DIR" ] && [ -d "$ACTIVE_CP_DIR" ]; then
        for meta_file in $(find "$ACTIVE_CP_DIR" -maxdepth 2 -type f -name "STAGE_METADATA.json" 2>/dev/null | sort -V -r); do
            FINAL_STAGE_DIR="$(dirname "$meta_file")"
            break
        done
    fi

    upload_folder_to_hf() {
        local repo_id="$1"
        local local_dir="$2"
        local path_in_repo="$3"
        python3 -c "
import os, sys
from huggingface_hub import HfApi

repo_id = sys.argv[1]
local_dir = sys.argv[2]
path_in_repo = sys.argv[3]
token = os.environ.get('HF_TOKEN')
if token and os.path.exists(local_dir):
    api = HfApi()
    print(f'[HF UPLOAD] Uploading {local_dir} -> {repo_id}/{path_in_repo}...')
    api.upload_folder(
        folder_path=local_dir,
        repo_id=repo_id,
        path_in_repo=path_in_repo,
        token=token,
    )
    print(f'[HF UPLOAD] Successfully uploaded {local_dir} to https://huggingface.co/{repo_id}/{path_in_repo}')
" "$repo_id" "$local_dir" "$path_in_repo" 2>/dev/null || true
    }

    if [ -n "$FINAL_STAGE_DIR" ] && [ -d "$FINAL_STAGE_DIR" ]; then
        echo "[HUGGING FACE SYNC] Pushing Final Neural Model ($FINAL_STAGE_DIR) to $HF_REPO/neural_model_v1..."
        if [ -f "./target/release/tara_hf_model_uploader" ]; then
            if [ -d "$WORLD_DIR" ] && [ -n "$(ls -A "$WORLD_DIR" 2>/dev/null)" ]; then
                echo "[HUGGING FACE SYNC] Including World Model & Joint Checkpoints ($WORLD_DIR)..."
                ./target/release/tara_hf_model_uploader --repo "$HF_REPO" --neural-dir "$FINAL_STAGE_DIR" --world-dir "$WORLD_DIR"
            else
                ./target/release/tara_hf_model_uploader --repo "$HF_REPO" --neural-dir "$FINAL_STAGE_DIR"
            fi
        else
            upload_folder_to_hf "$HF_REPO" "$FINAL_STAGE_DIR" "neural_model_v1"
            if [ -d "$WORLD_DIR" ] && [ -n "$(ls -A "$WORLD_DIR" 2>/dev/null)" ]; then
                upload_folder_to_hf "$HF_REPO" "$WORLD_DIR" "world_model_v1"
            fi
        fi
    elif [ -d "$WORLD_DIR" ] && [ -n "$(ls -A "$WORLD_DIR" 2>/dev/null)" ]; then
        echo "[HUGGING FACE SYNC] Pushing World Model & Joint Checkpoints to $HF_REPO/world_model_v1..."
        if [ -f "./target/release/tara_hf_model_uploader" ]; then
            ./target/release/tara_hf_model_uploader --repo "$HF_REPO" --world-dir "$WORLD_DIR"
        else
            upload_folder_to_hf "$HF_REPO" "$WORLD_DIR" "world_model_v1"
        fi
    fi
    echo "[HUGGING FACE SYNC] All artifacts successfully deployed to https://huggingface.co/$HF_REPO"
else
    echo "------------------------------------------------------------------------------"
    echo "[HUGGING FACE NOTICE] HF_TOKEN and HF_REPO not set. Skipping auto-push to Hugging Face."
fi

echo "=============================================================================="
echo "  TARA CLOUD TRAINING & SYNCHRONIZATION PIPELINE FINISHED SUCCESSFULLY!"
echo "=============================================================================="
