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

# 1. Enforce Cloud Training Environment Variables for Rust Orchestrator
export TARA_CLOUD_TRAINING=1
export TARA_MAX_MEMORY_MB="${TARA_MAX_MEMORY_MB:-32768}"
export GDRIVE_CHECKPOINT_DIR="${GDRIVE_CHECKPOINT_DIR:-/content/drive/MyDrive/TARA_CHECKPOINTS}"
export HF_DATASET_REPO="${HF_DATASET_REPO:-${HF_MODEL_REPO:-${HF_REPO:-manjukp6/tara}}}"
export HF_MODEL_REPO="${HF_MODEL_REPO:-${HF_DATASET_REPO}}"

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

# Ensure huggingface-cli is always available so downloads/uploads never fail silently
if ! command -v huggingface-cli >/dev/null 2>&1; then
    echo "[SETUP] Installing huggingface_hub CLI..."
    pip install -q -U "huggingface_hub[cli]"
fi

# Ensure NVIDIA driver and CUDA libraries are discoverable in Colab / Linux containers
export LD_LIBRARY_PATH="/usr/lib64-nvidia:/usr/local/cuda/compat:/usr/local/cuda-12.8/compat:${LD_LIBRARY_PATH:-}"

# Helper function for robust streaming downloads from Hugging Face
download_hf() {
    local repo_id="$1"
    local repo_type="$2"
    local pattern="$3"
    local local_dir="$4"
    python3 -c "
import sys, os
from huggingface_hub import snapshot_download
token = os.environ.get('HF_TOKEN') or None
try:
    snapshot_download(
        repo_id='$repo_id',
        repo_type='$repo_type',
        allow_patterns=['$pattern'],
        local_dir='$local_dir',
        token=token
    )
    sys.exit(0)
except Exception as e:
    print(f'Download error: {e}', file=sys.stderr)
    sys.exit(1)
"
}

# Authenticate upfront if HF_TOKEN is present
if [ -n "${HF_TOKEN:-}" ]; then
    echo "[AUTH] Authenticating with Hugging Face Hub..."
    python3 -c "
import huggingface_hub, os
try:
    huggingface_hub.login(token=os.environ['HF_TOKEN'], add_to_git_credential=True)
    print('Hugging Face Authentication Successful!')
except Exception as e:
    print('Hugging Face Login Notice:', e)
" || true
fi

# 2. Hardware Inspection & Upfront Binary Compilation
echo "------------------------------------------------------------------------------"
echo "[HARDWARE] Inspecting Cloud GPU Acceleration..."
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
else
    echo "[HARDWARE] No GPU detected. Defaulting to High-Performance Cloud CPU."
fi
echo "------------------------------------------------------------------------------"

echo "[BUILD] Compiling TARA Native Rust Binaries (Release Mode)..."
cargo build --release \
    --bin tara_pretraining_gate \
    --bin tara_training_stages \
    --bin manual_trainer \
    --bin train_candidate \
    --bin tara_model_sharder

# Dynamic Parameter Configuration
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
STAGE_MAX_STEPS="${STAGE_MAX_STEPS:-200}"
JOINT_STEPS="${JOINT_STEPS:-200}"
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

# Verify Neural Model Candidate weights exist on disk
NEURAL_SHARDS=(storage/models/tara_candidate_v1/*.safetensors)
if [ ! -e "${NEURAL_SHARDS[0]}" ]; then
    echo "=== [MODEL INGESTION: STREAMING NEURAL MODEL V1 FROM HUGGING FACE] ==="
    echo "[HF DOWNLOAD] Downloading neural_model_v1/* from https://huggingface.co/$HF_MODEL_SOURCE..."
    if download_hf "$HF_MODEL_SOURCE" "model" "neural_model_v1/*" "storage/models"; then
        if [ -d "storage/models/neural_model_v1" ]; then
            shopt -s nullglob
            mv storage/models/neural_model_v1/* storage/models/tara_candidate_v1/
            shopt -u nullglob
            rmdir storage/models/neural_model_v1
        fi
        CHECK_SHARDS=(storage/models/tara_candidate_v1/*.safetensors)
        if [ -e "${CHECK_SHARDS[0]}" ]; then
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
if [ ! -e "${WORLD_SHARDS[0]}" ]; then
    echo "=== [MODEL INGESTION: STREAMING WORLD MODEL V1 FROM HUGGING FACE] ==="
    echo "[HF DOWNLOAD] Downloading world_model_v1/* from https://huggingface.co/$HF_MODEL_SOURCE..."
    if download_hf "$HF_MODEL_SOURCE" "model" "world_model_v1/*" "storage/models"; then
        if [ -d "storage/models/world_model_v1" ]; then
            shopt -s nullglob
            mv storage/models/world_model_v1/* storage/models/tara_world_model/
            shopt -u nullglob
            rmdir storage/models/world_model_v1
        fi
        CHECK_WORLD=(storage/models/tara_world_model/*.safetensors)
        if [ -e "${CHECK_WORLD[0]}" ]; then
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

# 6. Execute Autonomous Extensible Multi-Stage Training Pipeline (Stages 0..5)
# Fallback scale_dir to MAIN_DS_DIR if storage/datasets/main_dataset_v1 is not present
SCALE_DS_DIR="storage/datasets/main_dataset_v1"
if [ ! -d "$SCALE_DS_DIR" ] || [ -z "$(ls -A "$SCALE_DS_DIR" 2>/dev/null)" ]; then
    SCALE_DS_DIR="$MAIN_DS_DIR"
fi

echo "=== [AUTONOMOUS EXTENSIBLE MULTI-STAGE TRAINING PIPELINE] ==="
./target/release/tara_training_stages \
    --all-stages \
    --checkpoints-dir "$ACTIVE_CP_DIR" \
    --main-dataset-dir "$MAIN_DS_DIR" \
    --scale-dir "$SCALE_DS_DIR" \
    --device auto \
    --precision auto \
    --batch-size "$TRAIN_BATCH_SIZE" \
    --max-steps "$STAGE_MAX_STEPS"

echo "Autonomous Multi-Stage Training Pipeline Completed Successfully!"

# 7. Execute Neural & World Model Joint Co-Training
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
    if [ -d "$ACTIVE_CP_DIR" ]; then
        for meta_file in $(find "$ACTIVE_CP_DIR" -maxdepth 2 -type f -name "STAGE_METADATA.json" 2>/dev/null | sort -V -r); do
            FINAL_STAGE_DIR="$(dirname "$meta_file")"
            break
        done
    fi

    if [ -n "$FINAL_STAGE_DIR" ] && [ -d "$FINAL_STAGE_DIR" ]; then
        echo "[HUGGING FACE SYNC] Pushing Final Neural Model ($FINAL_STAGE_DIR) to $HF_REPO/neural_model_v1..."
        huggingface-cli upload "$HF_REPO" "$FINAL_STAGE_DIR" "neural_model_v1" --repo-type model
    fi

    WORLD_DIR="manual_training/checkpoints"
    if [ -d "$WORLD_DIR" ] && [ -n "$(ls -A "$WORLD_DIR" 2>/dev/null)" ]; then
        echo "[HUGGING FACE SYNC] Pushing World Model & Joint Checkpoints to $HF_REPO/world_model_v1..."
        huggingface-cli upload "$HF_REPO" "$WORLD_DIR" "world_model_v1" --repo-type model
    fi
    echo "[HUGGING FACE SYNC] All artifacts successfully deployed to https://huggingface.co/$HF_REPO"
else
    echo "------------------------------------------------------------------------------"
    echo "[HUGGING FACE NOTICE] HF_TOKEN and HF_REPO not set. Skipping auto-push to Hugging Face."
fi

echo "=============================================================================="
echo "  TARA CLOUD TRAINING & SYNCHRONIZATION PIPELINE FINISHED SUCCESSFULLY!"
echo "=============================================================================="
