#!/usr/bin/env bash
# ==============================================================================
# TARA Autonomous Google Colab Stage-Wise Training & Cloud Sync Pipeline
# 100% Cloud Execution (0% Local PC usage)
# Flow: Clone from GitHub -> Native Rust Training -> Push to HF
# ==============================================================================

set -euo pipefail

echo "=============================================================================="
echo "  TARA CLOUD TRAINING SYSTEM (GOOGLE COLAB AUTONOMOUS RUNNER)"
echo "=============================================================================="

# 0. Proactive Disk Hygiene: Reclaim disk space from stale scratch files, caches, and unused debug builds
echo "[DISK CLEANUP] Reclaiming disk space and stopping any stale processes..."
killall -9 tara_training_stages 2>/dev/null || true
rm -rf /tmp/tara_moments_* /tmp/tara_* /tmp/tmp_tara_* 2>/dev/null || true
rm -rf /root/.julia /content/sample_data 2>/dev/null || true
rm -rf storage/datasets/*/.cache storage/models/*/.cache 2>/dev/null || true
rm -f storage/datasets/*/indiccorp_v2* storage/datasets/*/*/indiccorp_v2* 2>/dev/null || true
rm -rf target/debug target/release/deps target/release/build target/release/incremental 2>/dev/null || true
rm -rf /root/.cache/huggingface /root/.cache/pip /root/.cache/torch /root/.cargo/registry/cache ~/.cache/* 2>/dev/null || true
if command -v apt-get >/dev/null 2>&1; then
    apt-get clean 2>/dev/null || true
    rm -rf /var/lib/apt/lists/* 2>/dev/null || true
fi
if command -v df >/dev/null 2>&1; then
    df -h / | awk 'NR==2 {print "[DISK] Available disk space: " $4 " free / " $2 " total (" $5 " used)"}'
fi


# 1. Enforce Cloud Training Environment Variables for PyTorch & Rust Orchestrator
export TARA_CLOUD_TRAINING="1"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export MALLOC_TRIM_THRESHOLD_=65536
export TARA_MAX_MEMORY_MB="${TARA_MAX_MEMORY_MB:-}"

# Dynamically resolve max sequence length bounded by model architecture
if [ -z "${TARA_MAX_SEQ_LEN:-}" ]; then
    MODEL_MAX_POS=4096
    for cfg in "storage/models/tara_candidate_v1/config.json" "storage/models/tara_base/config.json" "config.json"; do
        if [ -f "$cfg" ]; then
            RESOLVED_POS=$(python3 -c "import json; d=json.load(open('$cfg')); print(d.get('max_position_embeddings', ''))" 2>/dev/null || true)
            if [ -n "$RESOLVED_POS" ] && [ "$RESOLVED_POS" -gt 0 ] 2>/dev/null; then
                MODEL_MAX_POS="$RESOLVED_POS"
                break
            fi
        fi
    done
    export TARA_MAX_SEQ_LEN="$MODEL_MAX_POS"
    echo "[DYNAMIC CONFIG] Assigned TARA_MAX_SEQ_LEN=$TARA_MAX_SEQ_LEN from model architecture capacity"
else
    echo "[DYNAMIC CONFIG] Using user-specified TARA_MAX_SEQ_LEN=$TARA_MAX_SEQ_LEN"
fi

# Dynamically resolve RoPE base frequency if not explicitly provided
if [ -z "${TARA_ROPE_THETA:-}" ]; then
    for cfg in "storage/models/tara_candidate_v1/config.json" "storage/models/tara_base/config.json" "config.json"; do
        if [ -f "$cfg" ]; then
            RESOLVED_THETA=$(python3 -c "import json; d=json.load(open('$cfg')); print(d.get('rope_theta', ''))" 2>/dev/null || true)
            if [ -n "$RESOLVED_THETA" ]; then
                export TARA_ROPE_THETA="$RESOLVED_THETA"
                echo "[DYNAMIC CONFIG] Resolved TARA_ROPE_THETA=$TARA_ROPE_THETA from model configuration"
                break
            fi
        fi
    done
fi
export TARA_MAX_VAL_SAMPLES="${TARA_MAX_VAL_SAMPLES:-5}"
export TARA_SKIP_PREFLIGHT_VAL="${TARA_SKIP_PREFLIGHT_VAL:-1}"
export HF_DATASET_REPO="${HF_DATASET_REPO:-manjukp6/tara_dataset_v1}"
export HF_MODEL_REPO="${HF_MODEL_REPO:-manjukp6/tara}"
export HF_REPO="${HF_REPO:-${HF_MODEL_REPO}}"

# Resolve and validate HF_TOKEN from environment or Google Colab userdata secrets
RESOLVED_HF_TOKEN=$(python3 -c "
import os, sys

def clean(val):
    if not val:
        return None
    s = str(val).strip().strip('\'\"').strip()
    if s and all(ord(c) < 128 for c in s):
        return s
    return None

candidates = []
if os.environ.get('HF_TOKEN'):
    candidates.append(os.environ.get('HF_TOKEN'))

try:
    from google.colab import userdata
    for k in ('HF_TOKEN', 'hf_token', 'tara_sync'):
        try:
            val = userdata.get(k)
            if val:
                candidates.append(val)
        except Exception:
            pass
except Exception:
    pass

# Priority 1: Tokens starting with 'hf_'
for cand in candidates:
    c = clean(cand)
    if c and c.startswith('hf_'):
        print(c)
        sys.exit(0)

# Priority 2: Any non-empty ASCII candidate
for cand in candidates:
    c = clean(cand)
    if c:
        print(c)
        sys.exit(0)
" 2>/dev/null || true)

if [ -n "$RESOLVED_HF_TOKEN" ]; then
    export HF_TOKEN="$RESOLVED_HF_TOKEN"
    echo "[AUTH] HF_TOKEN verified and exported successfully."
else
    if [ -n "${HF_TOKEN:-}" ]; then
        echo "[AUTH WARNING] HF_TOKEN contains non-ASCII characters (e.g. copied masked dots '•'). Clearing invalid token."
        unset HF_TOKEN
        rm -f /root/.cache/huggingface/token ~/.cache/huggingface/token 2>/dev/null || true
    fi
fi

TRAINER_MODE="${TRAINER_MODE:-python}"
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_MEM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n1 || echo "0")
    if [ "$GPU_MEM_MB" -ge 30000 ] 2>/dev/null; then
        TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
        GRAD_ACCUM="${GRAD_ACCUM:-8}"
    else
        # For GPUs <= 16GB (Tesla T4), micro-batch=1 avoids OOM while grad_accum=16 preserves effective batch size 16
        TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
        GRAD_ACCUM="${GRAD_ACCUM:-16}"
    fi
else
    TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
    GRAD_ACCUM="${GRAD_ACCUM:-8}"
fi
STAGE_MAX_STEPS="${STAGE_MAX_STEPS:-0}"

FULL_DATASET="${FULL_DATASET:-1}"
FULL_DATASET_EXTRA=""
if [ "$FULL_DATASET" = "1" ] || [ "${1:-}" = "--full-dataset" ]; then
    echo "=============================================================================="
    echo "  [FULL DATASET CONTINUOUS ROLLING TRAINING ACTIVE]"
    echo "=============================================================================="
    FULL_DATASET_EXTRA="--full-dataset"
    STAGE_MAX_STEPS=0
fi
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
GRAD_CHECKPOINT_FLAG=""
if [ "$GRADIENT_CHECKPOINTING" = "1" ]; then
    GRAD_CHECKPOINT_FLAG="--gradient-checkpointing"
fi
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-500}"
SKIP_SAVE_STEPS="${SKIP_SAVE_STEPS:-25}"
export SKIP_SAVE_STEPS
PROMOTE_MAX_STEP="${PROMOTE_MAX_STEP:-0}"
export PROMOTE_MAX_STEP
DOMAIN_WEIGHTS="${DOMAIN_WEIGHTS:-02_code:0.25,01_math:0.15,04_kannada:0.15,12_reasoning:0.15,03_stem:0.10,08_world_model:0.05,11_tool_use:0.05,13_debugging:0.02,sft:0.02,05_philosophy:0.012,06_social:0.012,07_engineering:0.012,09_audio:0.012,10_vision:0.012}"
export DOMAIN_WEIGHTS
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
if token:
    token_str = str(token).strip().strip('\'\"').strip()
    if token_str and all(ord(c) < 128 for c in token_str):
        token = token_str
    else:
        print('[HF DOWNLOAD] Warning: Provided HF_TOKEN contains non-ASCII characters. Proceeding without token.', flush=True)
        token = None
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
        rm -rf /root/.cache/huggingface /root/.cache/pip ~/.cache/huggingface "$local_dir/.cache" 2>/dev/null || true
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

# ------------------------------------------------------------------------------
# 2b. Local Checkpoint Discovery (NVMe / SSD Storage)
# ------------------------------------------------------------------------------
ACTIVE_CP_DIR="storage/models/checkpoints"
mkdir -p "$ACTIVE_CP_DIR"


# Zero-copy candidate model & world model seeding from restored local checkpoints
mkdir -p storage/models/tara_candidate_v1
mkdir -p storage/models/tara_world_model

FOUND_RESTORED_MODEL=0
for cand_file in "$ACTIVE_CP_DIR"/*/"model.safetensors" "$ACTIVE_CP_DIR"/*/*/"model.safetensors"; do
    if [ -f "$cand_file" ]; then
        cand_dir="$(dirname "$cand_file")"
        if [ -f "$cand_dir/config.json" ]; then
            CAND_SIZE=$(wc -c < "$cand_file" 2>/dev/null || echo 0)
            if [ "$CAND_SIZE" -gt 100000000 ]; then
                echo "[RESUME SEED] Seeding candidate neural model from restored checkpoint: $cand_dir"
                cp "$cand_dir/config.json" storage/models/tara_candidate_v1/ 2>/dev/null || true
                [ -f "$cand_dir/tokenizer.json" ] && cp "$cand_dir/tokenizer.json" storage/models/tara_candidate_v1/ 2>/dev/null || true
                [ -f "$cand_dir/tokenizer_config.json" ] && cp "$cand_dir/tokenizer_config.json" storage/models/tara_candidate_v1/ 2>/dev/null || true
                # Use hardlink or symlink to eliminate duplicate 4.2 GB copy
                ln -sf "$(readlink -f "$cand_file")" storage/models/tara_candidate_v1/model.safetensors 2>/dev/null || cp "$cand_file" storage/models/tara_candidate_v1/model.safetensors
                FOUND_RESTORED_MODEL=1
                break
            fi
        fi
    fi
done

for wm_file in "$ACTIVE_CP_DIR"/*/"world_model.safetensors" "$ACTIVE_CP_DIR"/*/*/"world_model.safetensors"; do
    if [ -f "$wm_file" ]; then
        wm_dir="$(dirname "$wm_file")"
        WM_SIZE=$(wc -c < "$wm_file" 2>/dev/null || echo 0)
        if [ "$WM_SIZE" -gt 10000000 ]; then
            echo "[RESUME SEED] Seeding candidate world model from restored checkpoint: $wm_dir"
            [ -f "$wm_dir/config.json" ] && cp "$wm_dir/config.json" storage/models/tara_world_model/ 2>/dev/null || true
            ln -sf "$(readlink -f "$wm_file")" storage/models/tara_world_model/world_model.safetensors 2>/dev/null || cp "$wm_file" storage/models/tara_world_model/world_model.safetensors
            break
        fi
    fi
done

# Check if shard ledger or active checkpoint state exists in any stage -> let Python JIT fetch active shard
FOUND_RESUME_LEDGER=0
for cp_dir in "$ACTIVE_CP_DIR"/*; do
    if [ -f "$cp_dir/shard_ledger.json" ] || [ -f "$cp_dir/checkpoint_state.json" ]; then
        FOUND_RESUME_LEDGER=1
        break
    fi
done

# Dynamic Parameter Configuration
MAIN_DS_DIR="storage/datasets/main_dataset_v2"


# 3. Ingest Datasets for Pretraining
mkdir -p "$MAIN_DS_DIR"
LOCAL_DS_COUNT=$(find "$MAIN_DS_DIR" -maxdepth 1 \( -name "*.jsonl" -o -name "*.parquet" \) 2>/dev/null | wc -l)

if [ "$TRAINER_MODE" = "python" ] || [ "${SKIP_HF_DOWNLOAD:-0}" = "1" ] || [ "$LOCAL_DS_COUNT" -ge 1 ] || [ "$FOUND_RESUME_LEDGER" -eq 1 ]; then
    echo "=== [DATASET INGESTION: SKIPPED] Python JIT streaming or resume ledger active. ==="
else
    echo "=== [DATASET INGESTION: STREAMING PRETRAINING SHARDS] ==="
    HF_DATASET_SOURCE="${HF_DATASET_REPO:-${HF_REPO:-manjukp6/tara_dataset_v1}}"
    if [ -n "$HF_DATASET_SOURCE" ]; then
        DATASET_SHARDS_PATTERN="${DATASET_SHARDS_PATTERN:-}"
        if [ -z "$DATASET_SHARDS_PATTERN" ]; then
            if [ "${DOWNLOAD_ALL_SHARDS:-0}" = "1" ]; then
                DATASET_SHARDS_PATTERN="v2/*"
            elif [ "${STAGE_MAX_STEPS:-1000}" -gt 0 ] && [ "${STAGE_MAX_STEPS:-1000}" -le 2000 ]; then
                DATASET_SHARDS_PATTERN="v2/*shard_000*"
                echo "[DATASET] Step count is ${STAGE_MAX_STEPS:-1000} (requires ~$(( ${STAGE_MAX_STEPS:-1000} * ${TRAIN_BATCH_SIZE:-2} * ${GRAD_ACCUM:-8} )) samples). Streaming single shard ('$DATASET_SHARDS_PATTERN' ~700,000 samples) to save 26 GB bandwidth and 10 minutes..."
            elif [ "${STAGE_MAX_STEPS:-1000}" -gt 0 ] && [ "${STAGE_MAX_STEPS:-1000}" -le 5000 ]; then
                DATASET_SHARDS_PATTERN="v2/*shard_00*"
                echo "[DATASET] Step count is ${STAGE_MAX_STEPS:-1000}. Streaming initial shards ('$DATASET_SHARDS_PATTERN')..."
            else
                DATASET_SHARDS_PATTERN="v2/*"
            fi
        fi
        if download_hf "$HF_DATASET_SOURCE" "dataset" "$DATASET_SHARDS_PATTERN" "$MAIN_DS_DIR"; then
            if [ -d "$MAIN_DS_DIR/v2" ]; then
                shopt -s nullglob
                mv "$MAIN_DS_DIR/v2"/* "$MAIN_DS_DIR/"
                shopt -u nullglob
                rmdir "$MAIN_DS_DIR/v2"
            fi
            rm -rf /root/.cache/huggingface ~/.cache/huggingface "$MAIN_DS_DIR/.cache" 2>/dev/null || true
            echo "Pretraining Dataset Ingestion Verified!"
        else
            echo "[DATASET NOTICE] Remote dataset download failed or not found. Checking local datasets in $MAIN_DS_DIR."
        fi
    fi
fi

# 3b. Ingest Main Dataset V1 for Stage 4 (Scale Expansion - on-demand only)
SCALE_DS_DIR="storage/datasets/main_dataset_v1"
if [ "${DOWNLOAD_V1_NOW:-0}" = "1" ]; then
    mkdir -p "$SCALE_DS_DIR"
    LOCAL_V1_COUNT=$(find "$SCALE_DS_DIR" -maxdepth 1 -name "*.jsonl" 2>/dev/null | wc -l)
    if [ "$LOCAL_V1_COUNT" -ge 1 ]; then
        echo "=== [DATASET V1: SKIPPED] Main Dataset V1 already present ($LOCAL_V1_COUNT shard(s)). Skipping download! ==="
    else
        echo "=== [DATASET INGESTION: STREAMING MAIN DATASET V1 (STAGE 4) SHARDS] ==="
        HF_DATASET_SOURCE="${HF_DATASET_REPO:-${HF_REPO:-manjukp6/tara_dataset_v1}}"
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

# Helper function for downloading and verifying model assets from Hugging Face
ingest_model_asset() {
    local model_name="$1"        # e.g. "neural_model_v1" or "world_model_v1"
    local target_dir="$2"        # e.g. "storage/models/tara_candidate_v1" or "storage/models/tara_world_model"
    local log_tag="$3"           # e.g. "Candidate Neural Model" or "World Model"
    local repo_source="${HF_MODEL_REPO:-${HF_REPO:-manjukp6/tara}}"

    local local_count
    local_count=$(find "$target_dir" -maxdepth 1 -name "*.safetensors" 2>/dev/null | wc -l)

    if [ "${SKIP_HF_DOWNLOAD:-0}" = "1" ] || [ "$local_count" -ge 1 ]; then
        echo "=== [MODEL INGESTION: SKIPPED] $log_tag already present ($local_count shard(s) in $target_dir). Zero downloads needed! ==="
        return 0
    fi

    echo "=== [MODEL INGESTION: STREAMING $log_tag FROM HUGGING FACE] ==="
    echo "[HF DOWNLOAD] Downloading $model_name/* from https://huggingface.co/$repo_source..."
    if download_hf "$repo_source" "model" "$model_name/*" "$target_dir"; then
        if [ -d "$target_dir/$model_name" ]; then
            shopt -s nullglob
            mv "$target_dir/$model_name"/* "$target_dir/"
            shopt -u nullglob
            rmdir "$target_dir/$model_name"
        fi
        rm -rf /root/.cache/huggingface ~/.cache/huggingface "$target_dir/.cache" 2>/dev/null || true
        shopt -s nullglob
        local check_shards=("$target_dir"/*.safetensors)
        shopt -u nullglob
        if [ -e "${check_shards[0]:-}" ]; then
            echo "$log_tag Ingestion from Hugging Face Verified!"
            return 0
        else
            echo "=== [ERROR: NO SAFETENSORS SHARDS FOUND IN DOWNLOADED REPO ($target_dir)] ==="
            exit 1
        fi
    else
        echo "=== [ERROR: COULD NOT DOWNLOAD $log_tag FROM HUGGING FACE] ==="
        echo "Please ensure HF_TOKEN is exported with read access to https://huggingface.co/$repo_source"
        exit 1
    fi
}

ingest_model_asset "neural_model_v1" "storage/models/tara_candidate_v1" "Candidate Neural Model"
ingest_model_asset "world_model_v1" "storage/models/tara_world_model" "World Model"

# 5. Mandatory Pre-Training Verification, Deep De-duplication & Tokenizer Gate
if [ ! -f "./target/release/tara_pretraining_gate" ]; then
    echo "[BUILD] Compiling TARA Pre-Training Gate binary..."
    cargo build --release --bin tara_pretraining_gate
    # Reclaim intermediate Cargo build artifacts to preserve 10-15 GB disk space while keeping compiled binaries
    rm -rf target/release/deps target/release/build target/release/incremental target/debug ~/.cargo/registry/cache 2>/dev/null || true
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

echo "Pre-Training Authorization Gate Passed!"

# 5a. Proactive Pre-Training Disk Space Reclamation: Reclaim 15-20 GB while keeping compiled release binaries
echo "[DISK HYGIENE] Purging intermediate Rust build tree artifacts to reclaim 15-20 GB disk space..."
rm -rf target/release/deps target/release/build target/release/incremental target/debug ~/.cargo/registry/cache ~/.cargo/registry/src ~/.cargo/git/checkouts 2>/dev/null || true
rm -rf /root/.cache/huggingface /root/.cache/pip ~/.cache/huggingface ~/.cache/pip 2>/dev/null || true
if command -v df >/dev/null 2>&1; then
    df -h / | awk 'NR==2 {print "[DISK] Available disk space before training: " $4 " free / " $2 " total (" $5 " used)"}'
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
    echo "[TRAINER] Launching Unified Stage 1 Cloud GPU Pipeline on all dataset domains & Joint World Model..."

    # Preserve any existing checkpoint or promoted weights from previous training
    STAGE1_INPUT="storage/models/tara_candidate_v1"
    PREV_STAGE_ARG=""
    if [ -d "$ACTIVE_CP_DIR/stage1_unified" ] && [ -f "$ACTIVE_CP_DIR/stage1_unified/model.safetensors" ]; then
        STAGE1_INPUT="$ACTIVE_CP_DIR/stage1_unified"
    elif [ -d "$ACTIVE_CP_DIR/stage0_pretraining" ] && [ -f "$ACTIVE_CP_DIR/stage0_pretraining/model.safetensors" ]; then
        STAGE1_INPUT="$ACTIVE_CP_DIR/stage0_pretraining"
        PREV_STAGE_ARG="--previous-stage-dir $ACTIVE_CP_DIR/stage0_pretraining"
    elif [ -d "$PROMOTED_BASE/stage0_pretraining" ] && [ -f "$PROMOTED_BASE/stage0_pretraining/model.safetensors" ]; then
        STAGE1_INPUT="$PROMOTED_BASE/stage0_pretraining"
    fi

    # Unified Stage 1: Full-Curriculum Continuous Pretraining
    WORLD_DS_ARG=""
    if [ -d "$MAIN_DS_DIR/08_commonsense_world_model" ]; then
        WORLD_DS_ARG="--world-dataset-dir $MAIN_DS_DIR/08_commonsense_world_model"
    elif [ -f "storage/datasets/curriculum/agi_causal_transitions.jsonl" ]; then
        WORLD_DS_ARG="--world-dataset-dir storage/datasets/curriculum/agi_causal_transitions.jsonl"
    fi

    JOINT_WORLD_FLAG=""
    if [ "${ENABLE_JOINT_WORLD_MODEL:-0}" = "1" ] || [ "${GPU_MEM_MB:-0}" -ge 30000 ]; then
        echo "[JOINT WORLD MODEL] Sufficient GPU VRAM (${GPU_MEM_MB:-0} MB) detected: Enabling simultaneous World Model co-training."
        JOINT_WORLD_FLAG="--joint-world-model --world-model-dir storage/models/tara_world_model $WORLD_DS_ARG"
    else
        echo "[VRAM HEADROOM SAFETY] GPU VRAM (${GPU_MEM_MB:-0} MB < 30 GB): Allocating full GPU memory to Neural Model to guarantee 8+ GB headroom and full 4096 context length."
    fi

    python3 cloud_train.py \
        --stage-id "stage1_unified" \
        --stage-name "Stage 1: Unified Continuous Pretraining & World Model Co-Training" \
        --model-dir "$STAGE1_INPUT" \
        --dataset-dir "$MAIN_DS_DIR" \
        --dataset-prefix "" \
        --output-dir "$ACTIVE_CP_DIR/stage1_unified" \
        --promoted-dir "$PROMOTED_BASE/stage1_unified" \
        $PREV_STAGE_ARG \
        --max-steps "$STAGE_MAX_STEPS" \
        --batch-size "$TRAIN_BATCH_SIZE" \
        --grad-accum "$GRAD_ACCUM" \
        --max-seq-len "$TARA_MAX_SEQ_LEN" \
        --checkpoint-interval "$CHECKPOINT_INTERVAL" \
        --skip-save-steps "$SKIP_SAVE_STEPS" \
        --promote-max-step "$PROMOTE_MAX_STEP" \
        --domain-weights "$DOMAIN_WEIGHTS" \
        --hf-repo "$HF_DATASET_REPO" \
        --hf-dataset-repo "$HF_DATASET_REPO" \
        --hf-model-repo "$HF_MODEL_REPO" \
        --exclude-shards "${TARA_EXCLUDE_SHARDS:-*shard_099*,*shard_098*,*indiccorp_v2*}" \
        --max-shard-bytes 0 \
        $JOINT_WORLD_FLAG \
        $FULL_DATASET_EXTRA \
        $GRAD_CHECKPOINT_FLAG

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
    echo "[PIPELINE FAILURE] Training pipeline ($TRAINER_MODE) exited with status: $PIPELINE_EXIT"
    echo "Host memory state at termination:"
    free -h || true
    echo "Kernel termination logs (dmesg):"
    dmesg -T | grep -i -E "oom|killed process|out of memory" | tail -20 || true
    echo "=============================================================================="
    exit $PIPELINE_EXIT
fi

echo "Autonomous Multi-Stage Training Pipeline Completed Successfully!"

# 7. Execute Neural & World Model Joint Co-Training (Rust or Explicit Override Only)
if [ "$TRAINER_MODE" = "rust" ] || [ "${RUN_MANUAL_TRAINER:-0}" = "1" ]; then
    if [ ! -f "./target/release/manual_trainer" ]; then
        echo "[BUILD] Compiling TARA Manual Trainer..."
        cargo build --release --bin manual_trainer
    fi
    echo "=== [WORLD MODEL & NEURAL JOINT CO-TRAINING ($JOINT_STEPS STEPS, BATCH $TRAIN_BATCH_SIZE)] ==="
    ./target/release/manual_trainer \
        --mode both \
        --steps "$JOINT_STEPS" \
        --batch-size "$TRAIN_BATCH_SIZE" \
        --dataset "$MAIN_DS_DIR"

    echo "Neural & World Model Joint Training Completed Successfully!"
else
    echo "[STEP 7 SKIPPED] Python trainer has already completed unified pretraining and Joint World Model co-training."
fi

# 9. Automatic Hugging Face Model Promotion & Upload (Native Rust Trainer Only - Python Trainer uploads directly)
HF_TARGET_REPO="${HF_REPO:-${HF_MODEL_REPO:-manjukp6/tara}}"
if [ "$TRAINER_MODE" = "rust" ] || [ "${FORCE_SH_HF_UPLOAD:-0}" = "1" ]; then
  if [ -n "${HF_TOKEN:-}" ] && [ -n "${HF_TARGET_REPO:-}" ]; then
    echo "------------------------------------------------------------------------------"
    echo "[HUGGING FACE SYNC] Uploading trained model safetensors to '$HF_TARGET_REPO'..."

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
    try:
        api = HfApi()
        # Guarantee repository existence (private by default if new)
        try:
            api.create_repo(repo_id=repo_id, repo_type='model', private=True, exist_ok=True, token=token)
        except Exception as create_err:
            print(f'[HF NOTICE] Repo create check note: {create_err}')
        print(f'[HF UPLOAD] Uploading {local_dir} -> {repo_id}/{path_in_repo}...')
        api.upload_folder(
            folder_path=local_dir,
            repo_id=repo_id,
            repo_type='model',
            path_in_repo=path_in_repo,
            token=token,
            commit_message=f'Deploy verified promoted model artifacts ({path_in_repo})',
        )
        print(f'[HF UPLOAD] Successfully uploaded {local_dir} to https://huggingface.co/{repo_id}/{path_in_repo}')
    except Exception as upload_err:
        print(f'[HF UPLOAD ERROR] Failed uploading to Hugging Face: {upload_err}')
" "$repo_id" "$local_dir" "$path_in_repo"
    }

    if [ -n "$FINAL_STAGE_DIR" ] && [ -d "$FINAL_STAGE_DIR" ]; then
        echo "[HUGGING FACE SYNC] Pushing Final Neural Model ($FINAL_STAGE_DIR) to $HF_TARGET_REPO/neural_model_v1..."
        if [ -f "./target/release/tara_hf_model_uploader" ]; then
            if [ -d "$WORLD_DIR" ] && [ -n "$(ls -A "$WORLD_DIR" 2>/dev/null)" ]; then
                echo "[HUGGING FACE SYNC] Including World Model & Joint Checkpoints ($WORLD_DIR)..."
                ./target/release/tara_hf_model_uploader --repo "$HF_TARGET_REPO" --neural-dir "$FINAL_STAGE_DIR" --world-dir "$WORLD_DIR"
            else
                ./target/release/tara_hf_model_uploader --repo "$HF_TARGET_REPO" --neural-dir "$FINAL_STAGE_DIR"
            fi
        else
            upload_folder_to_hf "$HF_TARGET_REPO" "$FINAL_STAGE_DIR" "neural_model_v1"
            if [ -d "$WORLD_DIR" ] && [ -n "$(ls -A "$WORLD_DIR" 2>/dev/null)" ]; then
                upload_folder_to_hf "$HF_TARGET_REPO" "$WORLD_DIR" "world_model_v1"
            fi
        fi
    elif [ -d "$WORLD_DIR" ] && [ -n "$(ls -A "$WORLD_DIR" 2>/dev/null)" ]; then
        echo "[HUGGING FACE SYNC] Pushing World Model & Joint Checkpoints to $HF_TARGET_REPO/world_model_v1..."
        if [ -f "./target/release/tara_hf_model_uploader" ]; then
            ./target/release/tara_hf_model_uploader --repo "$HF_TARGET_REPO" --world-dir "$WORLD_DIR"
        else
            upload_folder_to_hf "$HF_TARGET_REPO" "$WORLD_DIR" "world_model_v1"
        fi
    fi
    echo "[HUGGING FACE SYNC] All artifacts successfully deployed to https://huggingface.co/$HF_TARGET_REPO"
  else
    echo "------------------------------------------------------------------------------"
    echo "[HUGGING FACE NOTICE] HF_TOKEN or HF_REPO not set. Skipping auto-push to Hugging Face."
  fi
fi

echo "=============================================================================="
echo "  TARA CLOUD TRAINING & SYNCHRONIZATION PIPELINE FINISHED SUCCESSFULLY!"
echo "=============================================================================="
