#!/usr/bin/env bash
set -euo pipefail

# Unified World Model Training Launcher (Auto-routes to GPU or TPU)
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "[Launcher] Detected NVIDIA GPU accelerator. Invoking world_train_gpu.sh..."
    bash tara_training/world_model/world_train_gpu.sh "$@"
elif python -c "import torch_xla" 2>/dev/null; then
    echo "[Launcher] Detected Cloud TPU accelerator. Invoking world_train_tpu.sh..."
    bash tara_training/world_model/world_train_tpu.sh "$@"
else
    echo "[Launcher] Defaulting to CPU/GPU world training pipeline..."
    bash tara_training/world_model/world_train_gpu.sh "$@"
fi
