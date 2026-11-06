#!/usr/bin/env bash
# ==============================================================================
# TARA Autonomous Google Colab / Cloud GPU Training Runner
# 100% Standalone Execution - Modular Architecture
# ==============================================================================

set -euo pipefail

# 0. Git Environment & Filemode Protection
# Prevent git filemode permission changes (e.g. chmod +x) from causing dirty git status or merge conflicts
git config core.filemode false 2>/dev/null || true
if [ -d ".git" ]; then
    git config core.filemode false 2>/dev/null || true
fi

echo "=============================================================================="
echo "  TARA CLOUD TRAINING PIPELINE (AUTONOMOUS STANDALONE RUNNER)"
echo "=============================================================================="

# 0. Optional Disk Hygiene
if [ "${TARA_CLEAN_TRAINING_CACHES:-0}" = "1" ]; then
    echo "[DISK CLEANUP] Explicit cache cleanup enabled."
    rm -rf /tmp/tara_* /tmp/tmp_tara_* 2>/dev/null || true
    rm -rf /root/.cache/pip /root/.cache/torch /root/.cache/huggingface 2>/dev/null || true
    if command -v apt-get >/dev/null 2>&1; then
        apt-get clean 2>/dev/null || true
        rm -rf /var/lib/apt/lists/* 2>/dev/null || true
    fi
else
    echo "[DISK] Retaining package, model, and dataset caches for reuse. Set TARA_CLEAN_TRAINING_CACHES=1 to opt into cleanup."
fi
if command -v df >/dev/null 2>&1; then
    df -h / | awk 'NR==2 {print "[DISK] Available disk space: " $4 " free / " $2 " total (" $5 " used)"}'
fi

# 1. Environment Variables & CUDA Allocator Optimization
export TARA_CLOUD_TRAINING="1"
export PYTHONUNBUFFERED="1"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export MALLOC_TRIM_THRESHOLD_=65536
export HF_DATASET_REPO="${HF_DATASET_REPO:-${HF_REPO:-manjukp6/tara_dataset_v1}}"
export HF_MODEL_REPO="${HF_MODEL_REPO:-${HF_REPO:-manjukp6/tara}}"

# 2. Hardware Inspection & Dynamic VRAM-Aware Parameter Budgeting
echo "------------------------------------------------------------------------------"
echo "[HARDWARE] Inspecting Cloud GPU Acceleration..."
GPU_MEM_MB=0
GPU_NAME=""
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || echo "")
    GPU_MEM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d ' ' || echo "0")
else
    echo "[HARDWARE] No GPU detected. Defaulting to Cloud CPU execution."
fi
echo "------------------------------------------------------------------------------"

# Dynamic Batch Size & Gradient Accumulation based on physical VRAM
if [ -n "$GPU_MEM_MB" ] && [ "$GPU_MEM_MB" -ge 35000 ] 2>/dev/null; then
    # A100 / H100 (40GB / 80GB)
    DEFAULT_BATCH=8
    DEFAULT_ACCUM=4
    DEFAULT_OPTIM="adamw"
    DEFAULT_SEQ_LEN=4096
elif [ -n "$GPU_MEM_MB" ] && [ "$GPU_MEM_MB" -ge 20000 ] 2>/dev/null; then
    # L4 / A10G (24GB)
    DEFAULT_BATCH=4
    DEFAULT_ACCUM=8
    DEFAULT_OPTIM="adamw"
    DEFAULT_SEQ_LEN=4096
elif [ -n "$GPU_MEM_MB" ] && [ "$GPU_MEM_MB" -ge 12000 ] 2>/dev/null; then
    # Tesla T4 / V100 (15GB - 16GB) - Strict VRAM Safeguard: 2048 seq len saves 75% attention VRAM
    DEFAULT_BATCH=2
    DEFAULT_ACCUM=16
    DEFAULT_OPTIM="adamw8bit"
    DEFAULT_SEQ_LEN=2048
else
    # CPU / Low-spec
    DEFAULT_BATCH=1
    DEFAULT_ACCUM=32
    DEFAULT_OPTIM="adamw"
    DEFAULT_SEQ_LEN=1024
fi

TRAIN_BATCH_SIZE="${BATCH_SIZE:-$DEFAULT_BATCH}"
GRAD_ACCUM="${GRAD_ACCUM:-$DEFAULT_ACCUM}"
OPTIMIZER_TYPE="${OPTIMIZER:-$DEFAULT_OPTIM}"
LR="${LR:-3e-4}"
EPOCHS="${EPOCHS:-1}"
STAGE_MAX_STEPS="${STAGE_MAX_STEPS:-20000}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-100}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
GRAD_CHECKPOINT_FLAG=""
if [ "$GRADIENT_CHECKPOINTING" = "1" ]; then
    GRAD_CHECKPOINT_FLAG="--gradient-checkpointing"
fi
DOMAIN_WEIGHTS="${DOMAIN_WEIGHTS:-02_code:0.25,01_math:0.15,04_kannada:0.15,12_reasoning:0.15,03_stem:0.10,08_world_model:0.05,11_tool_use:0.05,13_debugging:0.02,sft:0.02,05_philosophy:0.012,06_social:0.012,07_engineering:0.012,09_audio:0.012,10_vision:0.012}"
export DOMAIN_WEIGHTS
export TARA_PREFETCH_BUFFER="${TARA_PREFETCH_BUFFER:-16}"
export TARA_RAM_SHARD_CACHE="${TARA_RAM_SHARD_CACHE:-1}"

echo "[DYNAMIC CONFIG] Batch Size: $TRAIN_BATCH_SIZE | Grad Accum: $GRAD_ACCUM | Optimizer: $OPTIMIZER_TYPE | LR: $LR | Grad Checkpointing: $GRADIENT_CHECKPOINTING"

# 3. Mount Google Drive if in Colab & Configure Checkpoint Sync
if [ -d "/content" ]; then
    echo "[GDRIVE] Checking Google Drive mount in Colab..."
    python3 -c "from gdrive_sync import mount_google_drive; mount_google_drive()" 2>/dev/null || true
    export GDRIVE_BACKUP_DIR="${GDRIVE_BACKUP_DIR:-/content/drive/MyDrive/tara_checkpoints}"
    mkdir -p "$GDRIVE_BACKUP_DIR"
fi

# 4. Resolve Hugging Face Authentication Token
RESOLVED_HF_TOKEN=$(python3 -c "
import os
candidates = []
if os.environ.get('HF_TOKEN'):
    candidates.append(os.environ.get('HF_TOKEN'))
try:
    from google.colab import userdata
    for k in ('HF_TOKEN', 'hf_token', 'tara_sync'):
        try:
            val = userdata.get(k)
            if val: candidates.append(val)
        except Exception: pass
except Exception: pass
print(candidates[0] if candidates else '')
" 2>/dev/null || true)

if [ -n "$RESOLVED_HF_TOKEN" ]; then
    export HF_TOKEN="$RESOLVED_HF_TOKEN"
    echo "[AUTH] HF_TOKEN resolved securely."
else
    echo "[AUTH] No HF_TOKEN detected; running in unauthenticated public mode."
fi

# 4b. Resolve Authoritative Root Public Key for Pre-Training Signature Gate
if [ -z "${TARA_AUTH_PUBLIC_KEY:-}" ]; then
    RESOLVED_PUBKEY=$(python3 -c "
import os, re
try:
    from google.colab import userdata
    k = userdata.get('TARA_AUTH_PUBLIC_KEY')
    if k and len(k.strip()) == 64:
        print(k.strip())
        exit(0)
except Exception:
    pass
for f in ('model_config/tara_auth_pubkey.hex', 'storage/vault/tara_auth_pubkey.hex'):
    if os.path.isfile(f):
        with open(f, 'r') as fp:
            c = fp.read().strip()
            if len(c) == 64 and re.fullmatch(r'[0-9a-fA-F]{64}', c):
                print(c)
                exit(0)
" 2>/dev/null || true)
    if [ -n "$RESOLVED_PUBKEY" ]; then
        export TARA_AUTH_PUBLIC_KEY="$RESOLVED_PUBKEY"
        echo "[SECURITY] Authoritative Root Ed25519 authorization public key resolved."
    fi
fi

# 4c. Production Ephemeral Key Safeguard (Fail-Closed)
if [ "${TARA_ALLOW_EPHEMERAL_KEY:-0}" = "1" ] || [ "${TARA_ALLOW_EPHEMERAL_KEY:-0}" = "true" ]; then
    echo "[SECURITY WARNING] TARA_ALLOW_EPHEMERAL_KEY was enabled! Disabling strictly for production pretraining..."
    unset TARA_ALLOW_EPHEMERAL_KEY
fi
export TARA_ALLOW_EPHEMERAL_KEY="0"

# 5. Install Dependencies if needed
if ! python3 -c "import safetensors, tokenizers, torch, bitsandbytes" 2>/dev/null; then
    echo "[SETUP] Installing required dependencies from requirements.txt..."
    pip install -q -r requirements.txt
fi

# 6. Dynamically resolve Max Sequence Length bounded by hardware VRAM budget
if [ -z "${TARA_MAX_SEQ_LEN:-}" ]; then
    MODEL_MAX_POS="${DEFAULT_SEQ_LEN:-2048}"
    for cfg in "storage/models/tara_candidate_v1/config.json" "storage/models/tara_base/config.json" "config.json"; do
        if [ -f "$cfg" ]; then
            RESOLVED_POS=$(python3 -c "import json; d=json.load(open('$cfg')); print(d.get('max_position_embeddings', ''))" 2>/dev/null || true)
            if [ -n "$RESOLVED_POS" ] && [ "$RESOLVED_POS" -gt 0 ] 2>/dev/null; then
                # If model architecture specifies lower capacity, honor it; otherwise cap to hardware budget
                if [ "$RESOLVED_POS" -lt "$MODEL_MAX_POS" ]; then
                    MODEL_MAX_POS="$RESOLVED_POS"
                fi
                break
            fi
        fi
    done
    export TARA_MAX_SEQ_LEN="$MODEL_MAX_POS"
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

# 7. Checkpoint Directory Setup
ACTIVE_CP_DIR="${ACTIVE_CP_DIR:-storage/models/checkpoints}"
mkdir -p "$ACTIVE_CP_DIR"

# 8. Multi-Source Checkpoint Auto-Discovery & Restore (Hugging Face, Google Drive, Local)
echo "[AUTO-RESTORE] Dynamically checking for highest step durable checkpoint across Hugging Face, GDrive, and Local..."
python3 -c "
import os, sys, shutil, json, re
from huggingface_hub import HfApi, snapshot_download

cp_base = '${ACTIVE_CP_DIR}'
cand_dir = 'storage/models/tara_candidate_v1'
stage0_dst = os.path.join(cp_base, 'stage0_pretraining')
os.makedirs(stage0_dst, exist_ok=True)
os.makedirs(cand_dir, exist_ok=True)

highest_step = -1
best_src = None
best_type = None

# 1. Local scan
def check_folder(d):
    global highest_step, best_src, best_type
    if not os.path.isdir(d):
        return
    s = -1
    st_file = os.path.join(d, 'checkpoint_state.json')
    if os.path.isfile(st_file):
        try:
            with open(st_file, 'r', encoding='utf-8') as f:
                s = int(json.load(f).get('step', -1))
        except Exception:
            pass
    if s < 0:
        m = re.search(r'step_(\d+)', d)
        if m:
            s = int(m.group(1))
    has_weights = os.path.isfile(os.path.join(d, 'model.safetensors')) or os.path.isfile(os.path.join(d, 'model.safetensors.index.json'))
    if s > highest_step and has_weights:
        highest_step = s
        best_src = d
        best_type = 'local'

check_folder(stage0_dst)
for root in ['tara_checkpoints/normal_checkpoint', 'tara_checkpoints/best_milestone']:
    if os.path.isdir(root):
        for item in os.listdir(root):
            check_folder(os.path.join(root, item))

# 2. GDrive scan
gdrive_dir = os.environ.get('GDRIVE_BACKUP_DIR', '/content/drive/MyDrive/tara_checkpoints')
if os.path.isdir(gdrive_dir):
    check_folder(gdrive_dir)
    for root in [os.path.join(gdrive_dir, 'normal_checkpoint'), os.path.join(gdrive_dir, 'best_milestone')]:
        if os.path.isdir(root):
            for item in os.listdir(root):
                check_folder(os.path.join(root, item))

# 3. Hugging Face scan
hf_repo = os.environ.get('HF_MODEL_REPO', 'manjukp6/tara')
hf_token = os.environ.get('HF_TOKEN')
if hf_repo:
    try:
        api = HfApi(token=hf_token)
        files = api.list_repo_files(hf_repo, repo_type='model')
        hf_steps = []
        for f in files:
            m = re.search(r'step_(\d+)', f)
            if m and ('normal_checkpoint' in f or 'best_milestone' in f):
                hf_steps.append((int(m.group(1)), f))
        if hf_steps:
            max_s, max_f = max(hf_steps, key=lambda x: x[0])
            if max_s > highest_step:
                highest_step = max_s
                best_src = max_f.split('/step_')[0] + f'/step_{highest_step}'
                best_type = 'hf'
    except Exception as e:
        print(f'[AUTO-RESTORE Notice] HF query note: {e}')

print(f'[AUTO-RESTORE] Highest pretraining checkpoint discovered: Step {highest_step} (Source: {best_type} -> {best_src})')

# If highest is on Hugging Face, download to stage0_dst
if best_type == 'hf' and best_src:
    print(f'[AUTO-RESTORE HF] Downloading Step {highest_step} from https://huggingface.co/{hf_repo}:{best_src}...')
    snapshot_download(
        repo_id=hf_repo,
        repo_type='model',
        local_dir='.',
        allow_patterns=[f'{best_src}/*'],
        token=hf_token,
        max_workers=4,
    )
    src_dir = best_src
    if os.path.isdir(src_dir):
        for f in os.listdir(src_dir):
            shutil.copy2(os.path.join(src_dir, f), os.path.join(stage0_dst, f))

# If highest is local/GDrive, sync to stage0_dst
elif best_src and best_src != stage0_dst:
    for f in os.listdir(best_src):
        s_file = os.path.join(best_src, f)
        if os.path.isfile(s_file):
            shutil.copy2(s_file, os.path.join(stage0_dst, f))

# Seed candidate dir
for f in os.listdir(stage0_dst):
    s_file = os.path.join(stage0_dst, f)
    d_file = os.path.join(cand_dir, f)
    if os.path.isfile(s_file) and not os.path.exists(d_file):
        try:
            os.link(s_file, d_file)
        except Exception:
            shutil.copy2(s_file, d_file)

print(f'[AUTO-RESTORE] Verified stage0_pretraining & candidate seeding complete at Step {highest_step}.')
"

# 9. Launch Autonomous Multi-Stage Pipeline Runner (Stages 0..5 extensible)
STAGES=("stage0_pretraining" "stage1_unified" "stage2_abilities" "stage3_continual" "stage4_scale" "stage5_frontier")
START_STAGE_IDX=0

# If specific stage is provided via argument, execute single stage; otherwise execute multi-stage loop
REQUESTED_STAGE="${1:-all}"
if [ "$REQUESTED_STAGE" != "all" ]; then
    echo "[START] Executing single requested stage: $REQUESTED_STAGE..."
    python3 -u train_main.py \
        --stage-id "$REQUESTED_STAGE" \
        --output-dir "$ACTIVE_CP_DIR/$REQUESTED_STAGE" \
        --batch-size "$TRAIN_BATCH_SIZE" \
        --grad-accum "$GRAD_ACCUM" \
        --optimizer "$OPTIMIZER_TYPE" \
        --lr "$LR" \
        --epochs "$EPOCHS" \
        --max-seq-len "$TARA_MAX_SEQ_LEN" \
        --max-steps "$STAGE_MAX_STEPS" \
        --checkpoint-interval "$CHECKPOINT_INTERVAL" \
        --domain-weights "$DOMAIN_WEIGHTS" \
        $GRAD_CHECKPOINT_FLAG \
        --resume \
        "$@"
    echo "[COMPLETE] Single stage $REQUESTED_STAGE finished successfully."
    exit 0
fi

echo "=============================================================================="
echo "  LAUNCHING AUTONOMOUS MULTI-STAGE TRAINING PIPELINE (STAGES 0..5)"
echo "=============================================================================="

for i in "${!STAGES[@]}"; do
    STAGE_ID="${STAGES[$i]}"
    STAGE_OUTPUT_DIR="$ACTIVE_CP_DIR/$STAGE_ID"
    STAGE_MODEL_DIR="storage/models/tara_candidate_v1"

    # Stage handoff: if prior stage produced a valid checkpoint, seed from prior stage output
    if [ "$i" -gt 0 ]; then
        PREV_STAGE_ID="${STAGES[$((i-1))]}"
        PREV_OUTPUT_DIR="$ACTIVE_CP_DIR/$PREV_STAGE_ID"
        if [ -f "$PREV_OUTPUT_DIR/model.safetensors" ]; then
            STAGE_MODEL_DIR="$PREV_OUTPUT_DIR"
            echo "[STAGE HANDOFF] Seeding $STAGE_ID candidate weights from previous stage ($PREV_STAGE_ID)..."
        fi
    fi

    echo "------------------------------------------------------------------------------"
    echo "[PIPELINE EXECUTION] Starting Stage $((i)): $STAGE_ID"
    echo "  Model Input Dir : $STAGE_MODEL_DIR"
    echo "  Output Dir      : $STAGE_OUTPUT_DIR"
    echo "  Batch Size      : $TRAIN_BATCH_SIZE (Accum: $GRAD_ACCUM)"
    echo "  Optimizer       : $OPTIMIZER_TYPE"
    echo "------------------------------------------------------------------------------"

    python3 -u train_main.py \
        --stage-id "$STAGE_ID" \
        --model-dir "$STAGE_MODEL_DIR" \
        --output-dir "$STAGE_OUTPUT_DIR" \
        --batch-size "$TRAIN_BATCH_SIZE" \
        --grad-accum "$GRAD_ACCUM" \
        --optimizer "$OPTIMIZER_TYPE" \
        --lr "$LR" \
        --epochs "$EPOCHS" \
        --max-seq-len "$TARA_MAX_SEQ_LEN" \
        --max-steps "$STAGE_MAX_STEPS" \
        --checkpoint-interval "$CHECKPOINT_INTERVAL" \
        --domain-weights "$DOMAIN_WEIGHTS" \
        $GRAD_CHECKPOINT_FLAG \
        --resume

    PIPELINE_STATUS=$?
    if [ $PIPELINE_STATUS -ne 0 ]; then
        echo "[FATAL ERROR] Stage $STAGE_ID exited with error code $PIPELINE_STATUS. Stopping pipeline."
        exit $PIPELINE_STATUS
    fi
    echo "[STAGE COMPLETE] Stage $STAGE_ID finished successfully."
done

echo "=============================================================================="
echo "  ALL STAGES (0..5) COMPLETED SUCCESSFULLY!"
echo "=============================================================================="
