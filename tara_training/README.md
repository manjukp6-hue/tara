# TARA Autonomous Cloud Training Engine (Modular & Standalone)

This directory contains the completely separated, modular, production-ready Python cloud training suite for TARA.
It is **100% standalone** and decoupled from local PC runtime crates, designed specifically for execution in Google Colab or Cloud GPU environments.

---

## Modular File Architecture

1. **`gdrive_sync.py`** (Google Drive & Checkpoint Synchronization)
   - Handles automated Google Drive mounting (`mount_google_drive`).
   - Disaster recovery & checkpoint mirroring to/from Google Drive.
   - Dynamic SHA-256 verification and atomic staging (`save_checkpoint`, `save_best_milestone`).
   - Proactive disk space hygiene, stale artifact pruning, and emergency reclamation.
   - Mid-way resumption and stage completion detection.

2. **`model_dataset.py`** (Model Architecture & Streaming Dataset)
   - Native PyTorch Causal Architecture (`TaraForCausalLM`, `TaraAttention`, `RMSNorm`, SwiGLU `TaraMlp`, RoPE).
   - Multi-Modal Encoders (`PyTorchVisionEncoder`, `PyTorchAudioEncoder`, `CrossModalProjector`).
   - Native Tokenizer (`TaraNativeTokenizer`, special token handling, byte fallback).
   - Streaming JSONL dataset engine (`TaraStreamingJsonlDataset`, JIT shard downloads).
   - Multi-domain weighted scheduling (`TaraMultiDomainScheduler`) and candidate loading.

3. **`train_main.py`** (Main Training Pipeline & Orchestration)
   - Multi-stage continuous training pipeline (Stages 0..N unbounded).
   - Live loss metrics: Total Loss, Perplexity (PPL), Cross-Entropy loss, Aux loss, LR, GradNorm, and Throughput.
   - Dynamic checkpoint triggers and model promotion to Hugging Face.
   - CLI parameter parser and main orchestrator.

4. **`verify_model.py`** (Architecture Compatibility Checker)
   - Standalone validator to verify that any exported checkpoint (`model.safetensors`, `config.json`, `tokenizer.json`) directly loads into TARA's native Rust engine (`tara_engine`) with zero errors.

5. **`run_train.sh`** (Cloud / Colab Startup Shell Script)
   - Sets environment variables, cleans disk caches, mounts GDrive, installs requirements, and runs `train_main.py`.

6. **`requirements.txt`**
   - Essential cloud dependencies (`torch`, `safetensors`, `tokenizers`, `huggingface_hub`).

---

## How to Run in Google Colab / Cloud

1. Upload the files in `tara_training` to your Google Colab instance or GitHub repo.
2. Run in a Colab bash cell or terminal:
   ```bash
   bash run_train.sh
   ```
   Or execute directly:
   ```bash
   python3 train_main.py --stage_id stage_0 --epochs 1 --batch_size 4
   ```

## Verification

To verify that any trained checkpoint is 100% compatible with TARA's native Rust architecture:
```bash
python3 verify_model.py --model_dir path/to/checkpoint
```
