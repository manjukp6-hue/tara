#!/usr/bin/env bash
# ==============================================================================
# TARA Autonomous Google Colab Stage-Wise Training & Cloud Sync Pipeline
# 100% Cloud Execution (0% Local PC usage)
# Flow: Clone from GitHub -> Native Rust Training -> Sync to Google Drive -> Push to HF
# ==============================================================================

set -e

echo "=============================================================================="
echo "  TARA CLOUD TRAINING SYSTEM (GOOGLE COLAB AUTONOMOUS RUNNER)"
echo "=============================================================================="

# 1. Install & Configure Rust Toolchain (Native Engine)
if ! command -v cargo >/dev/null 2>&1; then
    echo "[SETUP] Installing Native Rust Toolchain..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
else
    echo "[SETUP] Rust Toolchain already installed: $(cargo --version)"
fi

# Ensure cargo is on PATH
export PATH="$HOME/.cargo/bin:$PATH"

# 2. Hardware Inspection
echo "------------------------------------------------------------------------------"
echo "[HARDWARE] Inspecting Cloud GPU Acceleration..."
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
else
    echo "[HARDWARE] No GPU detected. Defaulting to High-Performance Cloud CPU."
fi
echo "------------------------------------------------------------------------------"

# 3. Execute Stage 1: Foundational Academic Curriculum Pretraining
echo "=== [STAGE 1: FOUNDATIONAL ACADEMIC CURRICULUM TRAINING] ==="
cargo run --release --bin tara_training_stages -- \
    --stage 1 \
    --device auto \
    --precision auto \
    --batch-size 4 \
    --epochs 1

echo "Stage 1 Completed Successfully!"

# 4. Execute Stage 2: Ability Acquisition & Self-Evolution Training
echo "=== [STAGE 2: ABILITY TRAINING & SELF-EVOLUTION] ==="
cargo run --release --bin tara_training_stages -- \
    --stage 2 \
    --device auto \
    --precision auto \
    --batch-size 4 \
    --epochs 1

echo "Stage 2 Completed Successfully!"

# 5. Execute Neural & World Model Joint Co-Training (Mode: Both)
echo "=== [WORLD MODEL & NEURAL JOINT CO-TRAINING (MODE: BOTH)] ==="
cargo run --release --bin manual_trainer -- \
    --mode both \
    --steps 100 \
    --batch-size 4 \
    --dataset storage/datasets/tara_dataset_filtered/canonical

echo "Neural & World Model Joint Training Completed Successfully!"

# 6. Checkpoint Preservation to Google Drive (Gmail Account Storage)
GDRIVE_DEST="/content/drive/MyDrive/TARA_CHECKPOINTS"
if [ -d "/content/drive/MyDrive" ]; then
    echo "------------------------------------------------------------------------------"
    echo "[GDRIVE BACKUP] Google Drive detected. Archiving all checkpoints (Neural + World Model)..."
    mkdir -p "$GDRIVE_DEST"
    cp -r storage/models/checkpoints/* "$GDRIVE_DEST/"
    if [ -d "manual_training/checkpoints" ]; then
        cp -r manual_training/checkpoints/* "$GDRIVE_DEST/"
    fi
    echo "[GDRIVE BACKUP] Successfully saved Neural and World Model checkpoints to: $GDRIVE_DEST"
else
    echo "[GDRIVE NOTICE] /content/drive/MyDrive not mounted. Checkpoints saved locally in Colab."
    echo "Tip: Run 'from google.colab import drive; drive.mount(\"/content/drive\")' to enable auto-sync."
fi

# 6. Automatic Hugging Face Model Upload
if [ -n "$HF_TOKEN" ]; then
    echo "------------------------------------------------------------------------------"
    echo "[HUGGING FACE SYNC] Uploading trained model safetensors to Hugging Face..."
    TARGET_REPO="${HF_REPO:-manjukp6/tara}"
    
    # Install huggingface_hub CLI if not present
    pip install -q -U huggingface_hub
    
    # Log in and upload
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
    
    # Upload Stage 2 working model (safetensors, config, tokenizer)
    STAGE2_DIR="storage/models/checkpoints/stage2_ability_training"
    if [ -d "$STAGE2_DIR" ]; then
        echo "[HUGGING FACE SYNC] Pushing Stage 2 Neural Model to $TARGET_REPO..."
        huggingface-cli upload "$TARGET_REPO" "$STAGE2_DIR" --repo-type model
    fi

    # Upload World Model and Joint Candidate Checkpoints
    WORLD_DIR="manual_training/checkpoints"
    if [ -d "$WORLD_DIR" ]; then
        echo "[HUGGING FACE SYNC] Pushing World Model & Joint Checkpoints to $TARGET_REPO..."
        huggingface-cli upload "$TARGET_REPO" "$WORLD_DIR" --repo-type model
    fi
    echo "[HUGGING FACE SYNC] All artifacts successfully deployed to https://huggingface.co/$TARGET_REPO"
else
    echo "------------------------------------------------------------------------------"
    echo "[HUGGING FACE NOTICE] HF_TOKEN not set. Skipping auto-push to Hugging Face."
    echo "Set export HF_TOKEN=\"hf_your_token\" and export HF_REPO=\"your_username/repo\" to enable."
fi

echo "=============================================================================="
echo "  TARA CLOUD TRAINING & SYNCHRONIZATION PIPELINE FINISHED SUCCESSFULLY!"
echo "=============================================================================="
