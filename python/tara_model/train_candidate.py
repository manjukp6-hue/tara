"""
python/tara_model/train_candidate.py

Controlled Training Engine for Candidate Model TARA-0.2-skills-aligned.
- STRICTLY PROTECTS storage/models/tara and TARA/MODEL/current_model.json (never touched).
- Initializes weights from baseline storage/models/tara/model.safetensors.
- Trains on verified unified dataset: storage/datasets/tara/train.jsonl (2,414 samples).
- Evaluates on canonical validation set: storage/datasets/tara/val.jsonl (262 samples).
- Enforces strict promotion baseline threshold (starting baseline val loss = 4.7526).
- Implements early stopping (patience=2) and small controlled time/epoch budget.
- Checkpoints best model to storage/models/TARA-0.2-skills-aligned/checkpoint_best.safetensors.
"""

import os
import sys
import json
import time
import math
import random
import hashlib
import argparse
from typing import List, Dict, Any, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from safetensors.torch import load_file, save_file

# Add project paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))

from tara_model.architecture import TaraForCausalLM, TaraConfig
from tara_model.tokenizer import TaraTokenizer


class TaraJsonlDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer: TaraTokenizer, max_len: int = 256):
        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                text = f"{d['prompt']} {d['completion']}"
                tokens = tokenizer.encode(text)
                if 1 < len(tokens) <= max_len:
                    self.samples.append(tokens)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# Compatibility alias


def collate_batch(batch: List[List[int]], pad_id: int = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_len = max(len(s) for s in batch)
    padded_inputs = []
    padded_labels = []
    attn_masks = []

    for s in batch:
        pad_len = max_len - len(s)
        inp = s + [pad_id] * pad_len
        lbl = s + [-100] * pad_len  # ignore index for loss
        mask = [1] * len(s) + [0] * pad_len

        padded_inputs.append(inp)
        padded_labels.append(lbl)
        attn_masks.append(mask)

    return (
        torch.tensor(padded_inputs, dtype=torch.long),
        torch.tensor(padded_labels, dtype=torch.long),
        torch.tensor(attn_masks, dtype=torch.long)
    )


def evaluate_model(model: nn.Module, val_loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for input_ids, labels, attention_mask in val_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            attention_mask = attention_mask.to(device)

            loss, _ = model(input_ids, labels=labels, attention_mask=attention_mask)
            # Count valid (non -100) target tokens
            valid_toks = (labels[..., 1:] != -100).sum().item()
            if valid_toks > 0:
                total_loss += loss.item() * valid_toks
                total_tokens += valid_toks

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("nan")
    ppl = math.exp(avg_loss) if avg_loss < 50 else float("inf")
    return avg_loss, ppl


def run_controlled_training(
    baseline_dir: str = "storage/models/tara",
    output_dir: str = "storage/models/TARA-0.2-skills-aligned",
    max_epochs: int = 6,
    batch_size: int = 32,
    learning_rate: float = 0.002,
    patience: int = 2,
    train_path: Optional[str] = None,
    val_path: Optional[str] = None,
    candidate_version_name: Optional[str] = None,
    max_duration_seconds: Optional[float] = None,
    custom_config: Optional[TaraConfig] = None,
    initial_state_dict: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    cand_name = candidate_version_name or os.path.basename(output_dir)
    print("=" * 70)
    print(f"   CONTROLLED TRAINING RUN: {cand_name}")
    print("=" * 70)

    # 1. Verification of Safe Output Paths
    assert os.path.abspath(output_dir) != os.path.abspath(baseline_dir), "FATAL: Output dir must not be baseline dir!"
    assert not output_dir.endswith("storage/models/tara"), "FATAL: Output dir must not target baseline model!"
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cpu")
    print(f"Device: {device}")

    # 2. Tokenizer & Datasets
    tokenizer = TaraTokenizer()
    if train_path is None:
        train_path = os.path.join(PROJECT_ROOT, "storage/datasets/tara/train.jsonl")
    if val_path is None:
        val_path = os.path.join(PROJECT_ROOT, "storage/datasets/tara/val.jsonl")

    print(f"Loading training dataset from {train_path}...")
    train_dataset = TaraJsonlDataset(train_path, tokenizer)
    val_dataset = TaraJsonlDataset(val_path, tokenizer)
    print(f"  -> Train samples: {len(train_dataset)}")
    print(f"  -> Val samples:   {len(val_dataset)}")

    pad_token_id = tokenizer.token_to_id.get("<|pad|>", 0)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_batch(b, pad_token_id)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_batch(b, pad_token_id)
    )

    # 3. Model Initialization from Baseline or Expanded Weights
    if custom_config is not None:
        cfg = custom_config
        cfg.version = cand_name
    else:
        config_path = os.path.join(baseline_dir, "config.json")
        cfg = TaraConfig.from_json_file(config_path)
        cfg.version = cand_name

    print(f"Initializing model architecture (vocab_size={cfg.vocab_size}, hidden={cfg.hidden_size}, layers={cfg.num_hidden_layers})...")
    model = TaraForCausalLM(cfg).to(device)

    if initial_state_dict is not None:
        print("Loading weights from supplied expanded state dictionary...")
        # Convert any python lists/numpy arrays to torch tensors if needed
        torch_state = {}
        for k, v in initial_state_dict.items():
            if isinstance(v, torch.Tensor):
                torch_state[k] = v.to(device)
            elif isinstance(v, list):
                torch_state[k] = torch.tensor(v, dtype=torch.float32, device=device)
            else:
                torch_state[k] = torch.as_tensor(v, device=device)
        model.load_state_dict(torch_state)
    else:
        baseline_weights_path = os.path.join(baseline_dir, "model.safetensors")
        print(f"Loading weights from protected baseline: {baseline_weights_path}...")
        baseline_state = load_file(baseline_weights_path)
        model.load_state_dict(baseline_state)

    # 4. Starting Baseline Standard
    print("Evaluating initial baseline validation loss before training...")
    init_val_loss, init_val_ppl = evaluate_model(model, val_loader, device)
    print(f"  -> Protected Baseline Validation Loss: {init_val_loss:.4f} (Perplexity: {init_val_ppl:.2f})")
    protected_baseline_threshold = init_val_loss

    # 5. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = max_epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=learning_rate * 0.1)

    # 6. Controlled Training Loop with Early Stopping
    history = []
    best_val_loss = protected_baseline_threshold
    best_epoch = 0
    epochs_without_improvement = 0
    start_time = time.time()

    print(f"\nStarting training loop (Max {max_epochs} epochs, patience={patience})...\n")

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_tokens = 0
        t0 = time.time()

        for batch_idx, (input_ids, labels, attention_mask) in enumerate(train_loader):
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            attention_mask = attention_mask.to(device)

            optimizer.zero_grad()
            loss, _ = model(input_ids, labels=labels, attention_mask=attention_mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            valid_toks = (labels[..., 1:] != -100).sum().item()
            if valid_toks > 0:
                epoch_loss += loss.item() * valid_toks
                epoch_tokens += valid_toks

        avg_train_loss = epoch_loss / epoch_tokens if epoch_tokens > 0 else float("nan")
        train_ppl = math.exp(avg_train_loss) if avg_train_loss < 50 else float("inf")

        # Evaluate Validation Loss
        val_loss, val_ppl = evaluate_model(model, val_loader, device)
        elapsed = time.time() - t0

        is_new_best = val_loss < best_val_loss
        status_flag = ""

        if is_new_best:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            status_flag = " [NEW BEST]"

            # Save best checkpoint
            best_ckpt_path = os.path.join(output_dir, "checkpoint_best.safetensors")
            save_file(model.state_dict(), best_ckpt_path)
        else:
            epochs_without_improvement += 1
            status_flag = f" [No improvement: {epochs_without_improvement}/{patience}]"

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "train_perplexity": round(train_ppl, 2),
            "val_loss": round(val_loss, 4),
            "val_perplexity": round(val_ppl, 2),
            "lr": round(scheduler.get_last_lr()[0], 6),
            "duration_s": round(elapsed, 2),
            "is_best": is_new_best
        }
        history.append(epoch_record)

        print(
            f"Epoch {epoch:2d}/{max_epochs:2d} | "
            f"Train Loss: {avg_train_loss:.4f} (PPL: {train_ppl:6.2f}) | "
            f"Val Loss: {val_loss:.4f} (PPL: {val_ppl:6.2f}) | "
            f"LR: {scheduler.get_last_lr()[0]:.6f} | "
            f"Time: {elapsed:.1f}s{status_flag}"
        )

        # Early stopping check
        if epochs_without_improvement >= patience:
            print(f"\n[EARLY STOPPING TRIGGERED] Validation loss did not improve for {patience} consecutive epochs.")
            break

        # Resource safety check: max duration budget
        if max_duration_seconds and (time.time() - start_time) >= max_duration_seconds:
            print(f"\n[RESOURCE SAFETY BUDGET EXCEEDED] Max training duration ({max_duration_seconds}s) reached.")
            break


    total_duration = time.time() - start_time
    print(f"\nTraining completed in {total_duration:.1f}s.")
    print(f"Best Checkpoint: Epoch {best_epoch} with Val Loss: {best_val_loss:.4f} (Baseline: {protected_baseline_threshold:.4f})")

    # 7. Finalize Best Checkpoint as Primary Candidate Artifact
    best_ckpt_path = os.path.join(output_dir, "checkpoint_best.safetensors")
    final_safetensors_path = os.path.join(output_dir, "model.safetensors")

    if os.path.exists(best_ckpt_path):
        # Load best weights into final model.safetensors
        best_state = load_file(best_ckpt_path)
        save_file(best_state, final_safetensors_path)
    else:
        save_file(model.state_dict(), final_safetensors_path)

    # Compute SHA-256
    with open(final_safetensors_path, "rb") as f:
        sha256 = hashlib.sha256(f.read()).hexdigest()

    # Save tokenizer artifacts
    with open(os.path.join(output_dir, "tokenizer.json"), "w", encoding="utf-8") as f:
        json.dump(tokenizer.export_to_dict(), f, indent=2, ensure_ascii=False)

    with open(os.path.join(output_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "tokenizer_class": "TaraTokenizer",
            "vocab_size": cfg.vocab_size,
            "model_max_length": 2048,
            "padding_side": "right"
        }, f, indent=2)

    with open(os.path.join(output_dir, "special_tokens_map.json"), "w", encoding="utf-8") as f:
        json.dump({
            "pad_token": "<|pad|>",
            "bos_token": "<|im_start|>",
            "eos_token": "<|im_end|>",
            "unk_token": "<|unk|>",
            "additional_special_tokens": [
                "<|creator_auth|>",
                "<|tara_rule|>",
                "<|tara_exec|>",
                "<|tara_skill|>",
                "<|tara_memory|>"
            ]
        }, f, indent=2)

    # Save config.json
    cfg_dict = cfg.to_dict()
    cfg_dict["version"] = cand_name
    total_model_params = sum(p.numel() for p in model.parameters())
    cfg_dict["total_parameters"] = total_model_params
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg_dict, f, indent=2)

    # Save training metadata
    training_meta = {
        "model_name": cand_name,
        "version": cand_name,
        "total_parameters": total_model_params,
        "checkpoint_sha256": sha256,
        "protected_baseline_threshold_loss": round(protected_baseline_threshold, 4),
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 4),
        "best_val_perplexity": round(math.exp(best_val_loss), 2) if best_val_loss < 50 else float("inf"),
        "baseline_loss_reduction_pct": round(((protected_baseline_threshold - best_val_loss) / protected_baseline_threshold) * 100, 2),
        "total_training_duration_seconds": round(total_duration, 2),
        "training_history": history,
        "status": "CONTROLLED_RUN_COMPLETED"
    }
    with open(os.path.join(output_dir, "training_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(training_meta, f, indent=2)

    print(f"\nSaved candidate model artifacts to: {output_dir}")
    print(f"SHA-256: {sha256}")
    return training_meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Controlled training of TARA candidate model.")
    parser.add_argument("--baseline_dir", type=str, default="storage/models/tara")
    parser.add_argument("--output_dir", type=str, default="storage/models/TARA-0.2-skills-aligned")
    parser.add_argument("--max_epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--train_path", type=str, default=None)
    parser.add_argument("--val_path", type=str, default=None)
    parser.add_argument("--candidate_version_name", type=str, default=None)
    args = parser.parse_args()

    run_controlled_training(
        baseline_dir=args.baseline_dir,
        output_dir=args.output_dir,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
        train_path=args.train_path,
        val_path=args.val_path,
        candidate_version_name=args.candidate_version_name
    )
