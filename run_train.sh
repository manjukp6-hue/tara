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

# 0. Proactive Disk Hygiene
echo "[DISK CLEANUP] Reclaiming disk space from caches and temporary files..."
rm -rf /tmp/tara_* /tmp/tmp_tara_* 2>/dev/null || true
rm -rf /root/.cache/pip /root/.cache/torch /root/.cache/huggingface/hub 2>/dev/null || true
if command -v apt-get >/dev/null 2>&1; then
    apt-get clean 2>/dev/null || true
    rm -rf /var/lib/apt/lists/* 2>/dev/null || true
fi
if command -v df >/dev/null 2>&1; then
    df -h / | awk 'NR==2 {print "[DISK] Available disk space: " $4 " free / " $2 " total (" $5 " used)"}'
fi

# 1. Environment Variables & CUDA Allocator Optimization
export TARA_CLOUD_TRAINING="1"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export MALLOC_TRIM_THRESHOLD_=65536
export HF_DATASET_REPO="${HF_DATASET_REPO:-${HF_REPO:-manjukp6/tara_dataset_v1}}"
export HF_MODEL_REPO="${HF_MODEL_REPO:-${HF_REPO:-manjukp6/tara}}"

# 2. Hardware Inspection & Dynamic VRAM-Aware Parameter Budgeting
echo "------------------------------------------------------------------------------"
echo "[HARDWARE] Inspecting Cloud GPU Acceleration..."
GPU_MEM_MB=0
GPU_NAME=""
GPU_COUNT=0
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || echo "")
    GPU_MEM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d ' ' || echo "0")
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo "1")
    GPU_COUNT=$(echo "$GPU_COUNT" | tr -d ' ')
else
    echo "[HARDWARE] No GPU detected. Defaulting to Cloud CPU execution."
fi
echo "------------------------------------------------------------------------------"

# Dynamic Batch Size & Gradient Accumulation based on physical VRAM and GPU count
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
    # Tesla T4 / V100 (15GB - 16GB)
    if [ "$GPU_COUNT" -ge 2 ] 2>/dev/null; then
        # Multi-GPU (e.g. Kaggle dual Tesla T4 x 2):
        # Per-GPU micro-batch: 1 per GPU * 2 GPUs * accum 16 = exactly 32 effective batch size.
        # Micro-batch 1 uses ~9.5 GB VRAM per GPU, leaving 5+ GB free headroom (Zero OOM Guaranteed).
        DEFAULT_BATCH=1
        DEFAULT_ACCUM=16
        echo "[HARDWARE] Multi-GPU ($GPU_COUNT x $GPU_NAME) detected. Configured Micro-Batch: $DEFAULT_BATCH (1 per GPU), Grad Accum: $DEFAULT_ACCUM."
    else
        # Single GPU (e.g. Google Colab free Tesla T4):
        # 1 GPU * batch 2 * accum 16 = 32 effective batch size.
        DEFAULT_BATCH=2
        DEFAULT_ACCUM=16
    fi
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
STAGE_MAX_STEPS="${STAGE_MAX_STEPS:-50000}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-100}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100}"
PROMOTE_INTERVAL="${PROMOTE_INTERVAL:-1000}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
GRAD_CHECKPOINT_FLAG=""
if [ "$GRADIENT_CHECKPOINTING" = "1" ]; then
    GRAD_CHECKPOINT_FLAG="--gradient-checkpointing"
fi
DOMAIN_WEIGHTS="${DOMAIN_WEIGHTS:-02_code:0.25,01_math:0.15,04_kannada:0.15,12_reasoning:0.15,03_stem:0.10,08_world_model:0.05,11_tool_use:0.05,13_debugging:0.02,sft:0.02,05_philosophy:0.012,06_social:0.012,07_engineering:0.012,09_audio:0.012,10_vision:0.012}"
export DOMAIN_WEIGHTS

echo "[DYNAMIC CONFIG] Batch Size: $TRAIN_BATCH_SIZE | Grad Accum: $GRAD_ACCUM | Optimizer: $OPTIMIZER_TYPE | LR: $LR | Grad Checkpointing: $GRADIENT_CHECKPOINTING | Eval Interval: $EVAL_INTERVAL | Promote Interval: $PROMOTE_INTERVAL"

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
for token_file in ('/root/.cache/huggingface/token', os.path.expanduser('~/.cache/huggingface/token')):
    if os.path.isfile(token_file):
        try:
            tok = open(token_file).read().strip()
            if tok: candidates.append(tok)
        except Exception: pass
try:
    from google.colab import userdata
    for k in ('tara_sync', 'HF_TOKEN', 'hf_token'):
        try:
            val = userdata.get(k)
            if val: candidates.append(val)
        except Exception: pass
except Exception: pass
print(candidates[0] if candidates else '')
" 2>/dev/null || true)

HF_TOKEN_FLAG=""
if [ -n "$RESOLVED_HF_TOKEN" ]; then
    export HF_TOKEN="$RESOLVED_HF_TOKEN"
    export HUGGING_FACE_HUB_TOKEN="$RESOLVED_HF_TOKEN"
    HF_TOKEN_FLAG="--hf-token $RESOLVED_HF_TOKEN"
    echo "[AUTH] HF_TOKEN resolved securely (length: ${#RESOLVED_HF_TOKEN})."
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

# 6b. Configure High-Performance DDP Launcher for Multi-GPU
if [ "$GPU_COUNT" -ge 2 ] 2>/dev/null; then
    PY_CMD="torchrun --nproc_per_node=$GPU_COUNT"
    echo "[LAUNCHER DDP] Multi-GPU ($GPU_COUNT x $GPU_NAME) detected. Using PyTorch DDP launcher: $PY_CMD"
else
    PY_CMD="python3"
    echo "[LAUNCHER] Single GPU / CPU detected. Using standard launcher: $PY_CMD"
fi

# 7. Checkpoint Directory Setup
ACTIVE_CP_DIR="${ACTIVE_CP_DIR:-storage/models/checkpoints}"
mkdir -p "$ACTIVE_CP_DIR"

# 8. Multi-Cloud Checkpoint Discovery & Auto-Restore (Google Drive vs Hugging Face vs Local)
echo "[MULTI-CLOUD RESTORE] Auditing Google Drive and Hugging Face for latest durable checkpoints..."
python3 -c "
from gdrive_sync import resolve_and_auto_select_checkpoint, resolve_gdrive_restore_dir, resolve_gdrive_backup_dir, check_model_weights_exist
import os, shutil

cp_base = '${ACTIVE_CP_DIR}'
cand_dir = 'storage/models/tara_candidate_v1'
dst = os.path.join(cp_base, 'stage0_pretraining')
gdir = resolve_gdrive_restore_dir() or resolve_gdrive_backup_dir()
hf_repo = os.environ.get('HF_MODEL_REPO', os.environ.get('HF_REPO', None))
hf_tok = os.environ.get('HF_TOKEN', None)

if not check_model_weights_exist(dst):
    resolve_and_auto_select_checkpoint(
        local_ckpt_dir=dst,
        stage_id='stage0_pretraining',
        gdrive_base_dir=gdir,
        hf_repo_id=hf_repo,
        hf_token=hf_tok,
    )

os.makedirs(cand_dir, exist_ok=True)
if check_model_weights_exist(dst) and not check_model_weights_exist(cand_dir):
    for f in os.listdir(dst):
        s = os.path.join(dst, f)
        d = os.path.join(cand_dir, f)
        if os.path.isfile(s) and not os.path.exists(d):
            try:
                os.link(s, d)
            except Exception:
                shutil.copy2(s, d)
    print(f'[MULTI-CLOUD RESTORE] Seeded candidate directory {cand_dir} from restored checkpoint.')
" 2>/dev/null || true

# 8b. Ensure candidate directory has baseline config and tokenizer from model_config if missing
mkdir -p storage/models/tara_candidate_v1
for f in config.json tokenizer.json tokenizer_config.json special_tokens_map.json; do
    if [ -f "model_config/$f" ] && [ ! -f "storage/models/tara_candidate_v1/$f" ]; then
        cp "model_config/$f" "storage/models/tara_candidate_v1/$f"
    fi
done

# 9. Launch Autonomous Multi-Stage Pipeline Runner (Stages 0..5 extensible)
STAGES=("stage0_pretraining" "stage1_unified" "stage2_abilities" "stage3_continual" "stage4_scale" "stage5_frontier")
START_STAGE_IDX=0

# If specific stage is provided via argument, execute single stage; otherwise execute multi-stage loop
REQUESTED_STAGE="${1:-all}"
if [ "$REQUESTED_STAGE" != "all" ]; then
    echo "[START] Executing single requested stage: $REQUESTED_STAGE..."
    $PY_CMD train_main.py \
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
        --eval-interval "$EVAL_INTERVAL" \
        --promote-interval "$PROMOTE_INTERVAL" \
        --domain-weights "$DOMAIN_WEIGHTS" \
        $GRAD_CHECKPOINT_FLAG \
        $HF_TOKEN_FLAG \
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
        if [ -f "$PREV_OUTPUT_DIR/model.safetensors" ] || [ -f "$PREV_OUTPUT_DIR/model.safetensors.index.json" ]; then
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

    $PY_CMD train_main.py \
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
        --eval-interval "$EVAL_INTERVAL" \
        --promote-interval "$PROMOTE_INTERVAL" \
        --domain-weights "$DOMAIN_WEIGHTS" \
        $GRAD_CHECKPOINT_FLAG \
        $HF_TOKEN_FLAG \
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
