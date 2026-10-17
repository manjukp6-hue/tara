#!/usr/bin/env python3
"""
TARA Autonomous Cloud Training Engine (Google Colab / Cloud GPU Runner)
=============================================================================
Strictly isolated to Cloud Pretraining only (per Project Governance Rule 15).
Ingests JSONL dataset shards, trains the 1B neural base model on Cloud GPU (FP16/BF16),
and exports standard `model.safetensors`, `config.json`, and `tokenizer.json`
for 100% native Rust inference and serving in `tara_engine`.
=============================================================================
"""

import argparse
import glob
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Optional imports with friendly fallback
try:
    from safetensors.torch import save_file, load_file
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

try:
    from tokenizers import Tokenizer
    HAS_TOKENIZERS = True
except ImportError:
    HAS_TOKENIZERS = False


# =============================================================================
# 1. Native PyTorch Tara Causal Architecture (Matches config.json 1-to-1)
# =============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


def precompute_rope_freqs_cis(dim: int, end: int, theta: float = 500000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def apply_rope(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # xq: [batch, seq_len, n_heads, head_dim]
    # xk: [batch, seq_len, n_kv_heads, head_dim]
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[:xq.shape[1], :].unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, head_dim // 2]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class TaraAttention(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.n_heads = config["num_attention_heads"]
        self.n_kv_heads = config.get("num_key_value_heads", self.n_heads)
        self.head_dim = config.get("head_dim", self.hidden_size // self.n_heads)
        self.n_rep = self.n_heads // self.n_kv_heads

        self.q_proj = nn.Linear(self.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        q = self.q_proj(x).view(bsz, seqlen, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        q, k = apply_rope(q, k, freqs_cis)

        # GQA repeat KV
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=2)
            v = v.repeat_interleave(self.n_rep, dim=2)

        # Transpose to [bsz, n_heads, seqlen, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Flash / SDPA scaled dot product attention
        scores = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=(mask is None and seqlen > 1)
        )

        out = scores.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.o_proj(out)


class TaraMlp(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        hs = config["hidden_size"]
        inter = config["intermediate_size"]
        self.gate_proj = nn.Linear(hs, inter, bias=False)
        self.up_proj = nn.Linear(hs, inter, bias=False)
        self.down_proj = nn.Linear(inter, hs, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class TaraDecoderLayer(nn.Module):
    def __init__(self, config: dict, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.input_layernorm = RMSNorm(config["hidden_size"], eps=config.get("rms_norm_eps", 1e-5))
        self.self_attn = TaraAttention(config)
        self.post_attention_layernorm = RMSNorm(config["hidden_size"], eps=config.get("rms_norm_eps", 1e-5))
        self.mlp = TaraMlp(config)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = x + self.self_attn(self.input_layernorm(x), freqs_cis, mask)
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out


class TaraModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.layers = nn.ModuleList([
            TaraDecoderLayer(config, i) for i in range(config["num_hidden_layers"])
        ])
        self.norm = RMSNorm(config["hidden_size"], eps=config.get("rms_norm_eps", 1e-5))

        # Precompute RoPE complex table
        head_dim = config.get("head_dim", config["hidden_size"] // config["num_attention_heads"])
        max_pos = config.get("max_position_embeddings", 4096)
        theta = config.get("rope_theta", 500000.0)
        self.register_buffer("freqs_cis", precompute_rope_freqs_cis(head_dim, max_pos, theta), persistent=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seqlen = input_ids.shape[1]
        h = self.embed_tokens(input_ids)
        freqs_cis = self.freqs_cis[:seqlen]

        for layer in self.layers:
            h = layer(h, freqs_cis)

        return self.norm(h)


class TaraForCausalLM(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.model = TaraModel(config)
        self.lm_head = nn.Linear(config["hidden_size"], config["vocab_size"], bias=False)

        # Tie weights if requested
        if config.get("tie_word_embeddings", False):
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        hidden_states = self.model(input_ids)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift for causal next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return logits, loss


# =============================================================================
# 2. Dataset Ingestion & Shard Loader
# =============================================================================

class TaraJsonlDataset(Dataset):
    def __init__(self, file_paths: List[str], tokenizer: Tokenizer, max_seq_len: int = 256):
        self.samples: List[Tuple[List[int], int]] = []
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        print(f"[Dataset] Loading records from {len(file_paths)} shard file(s)...")
        total_loaded = 0
        for fp in file_paths:
            if not os.path.exists(fp):
                continue
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        inp_text = record.get("input", "")
                        out_text = record.get("output", "")
                        if not inp_text and not out_text:
                            continue

                        # Tokenize
                        enc_inp = tokenizer.encode(inp_text).ids
                        enc_out = tokenizer.encode(out_text).ids
                        target_start = len(enc_inp)

                        full_ids = enc_inp + enc_out
                        if len(full_ids) > self.max_seq_len:
                            full_ids = full_ids[:self.max_seq_len]

                        if len(full_ids) >= 4:
                            self.samples.append((full_ids, target_start))
                            total_loaded += 1
                    except Exception:
                        continue

        print(f"[Dataset] Successfully ingested {len(self.samples)} valid sequence pairs.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ids, target_start = self.samples[idx]
        input_ids = torch.tensor(ids, dtype=torch.long)
        labels = input_ids.clone()
        # Mask out input tokens so loss is computed strictly on outputs
        eff_start = min(target_start, len(ids))
        labels[:eff_start] = -100
        return input_ids, labels


def collate_fn(batch):
    max_len = max(len(b[0]) for b in batch)
    bsz = len(batch)
    padded_inputs = torch.zeros(bsz, max_len, dtype=torch.long)
    padded_labels = torch.full((bsz, max_len), -100, dtype=torch.long)

    for i, (inp, lab) in enumerate(batch):
        l = len(inp)
        padded_inputs[i, :l] = inp
        padded_labels[i, :l] = lab

    return padded_inputs, padded_labels


# =============================================================================
# 3. Model Weight Loading & SafeTensors Exporter
# =============================================================================

def load_candidate_weights(model: nn.Module, model_dir: str):
    """Loads existing candidate safetensors weights if present, ensuring continuous progression."""
    safetensors_files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not safetensors_files or not HAS_SAFETENSORS:
        print("[Weights] No existing safetensors shards found or safetensors not installed. Initializing fresh weights.")
        return

    print(f"[Weights] Ingesting {len(safetensors_files)} existing SafeTensors shard(s)...")
    state_dict = {}
    for sf in safetensors_files:
        shard_tensors = load_file(sf, device="cpu")
        state_dict.update(shard_tensors)

    # Filter and remap into model
    model_sd = model.state_dict()
    matched = 0
    for k, v in state_dict.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            model_sd[k].copy_(v)
            matched += 1

    print(f"[Weights] Successfully restored {matched}/{len(model_sd)} parameter tensors from candidate.")


def export_checkpoint(model: nn.Module, config: dict, tokenizer_path: str, output_dir: str):
    """Exports trained weights as standard model.safetensors, config.json, tokenizer.json."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"[Export] Saving canonical production checkpoint to: {output_dir}")

    # 1. config.json
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # 2. tokenizer.json
    if os.path.exists(tokenizer_path):
        import shutil
        shutil.copy2(tokenizer_path, os.path.join(output_dir, "tokenizer.json"))

    # 3. model.safetensors
    state_dict = {k: v.contiguous().cpu() for k, v in model.state_dict().items()}
    if HAS_SAFETENSORS:
        save_file(state_dict, os.path.join(output_dir, "model.safetensors"))
        print("[Export] SafeTensors checkpoint written successfully.")
    else:
        torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))
        print("[Export] PyTorch model bin written successfully.")


# =============================================================================
# 4. Main Autonomous Training Loop
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="TARA Autonomous Cloud Training Pipeline")
    parser.add_argument("--model-dir", type=str, default="storage/models/tara_candidate_v1", help="Candidate model directory")
    parser.add_argument("--dataset-dir", type=str, default="storage/datasets/main_dataset_v2", help="Dataset directory")
    parser.add_argument("--output-dir", type=str, default="storage/models/checkpoints/stage0_pretraining", help="Output checkpoint directory")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Micro-batch size per device")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max-steps", type=int, default=200, help="Max optimizer steps")
    parser.add_argument("--max-seq-len", type=int, default=256, help="Max sequence length")
    args = parser.parse_args()

    print("=" * 78)
    print("  TARA CLOUD TRAINING SYSTEM (HIGH-THROUGHPUT GPU RUNNER)")
    print("=" * 78)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Hardware] Active Device: {device}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        print(f"[Hardware] GPU: {gpu_name} | VRAM: {vram_mb:.1f} MB")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Load configuration
    config_path = os.path.join(args.model_dir, "config.json")
    if not os.path.exists(config_path):
        print(f"[Error] Missing config.json at {config_path}")
        sys.exit(1)
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    print(f"[Architecture] Vocab: {config['vocab_size']}, Hidden: {config['hidden_size']}, Layers: {config['num_hidden_layers']}, Heads: {config['num_attention_heads']}")

    # Load Tokenizer
    tok_path = os.path.join(args.model_dir, "tokenizer.json")
    if not os.path.exists(tok_path) or not HAS_TOKENIZERS:
        print(f"[Error] tokenizer.json missing at {tok_path} or tokenizers package not installed.")
        sys.exit(1)
    tokenizer = Tokenizer.from_file(tok_path)
    print(f"[Tokenizer] Loaded vocabulary from {tok_path}")

    # Build Model
    model = TaraForCausalLM(config).to(device)
    load_candidate_weights(model, args.model_dir)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] Total parameters: {total_params:,} ({total_params * 4 / 1e6:.1f} MB in FP32)")

    # Find JSONL shards
    shard_files = sorted(glob.glob(os.path.join(args.dataset_dir, "*.jsonl")))
    if not shard_files:
        # Fallback to main_dataset_v1 if v2 is empty
        shard_files = sorted(glob.glob("storage/datasets/main_dataset_v1/*.jsonl"))
    if not shard_files:
        print(f"[Error] No JSONL shards found in {args.dataset_dir} or fallback.")
        sys.exit(1)

    dataset = TaraJsonlDataset(shard_files[:10], tokenizer, max_seq_len=args.max_seq_len)
    if len(dataset) == 0:
        print("[Error] Ingested 0 dataset samples. Aborting.")
        sys.exit(1)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    # Optimizer with Fused AdamW where supported
    fused_available = 'fused' in torch.optim.AdamW.__init__.__code__.co_varnames and torch.cuda.is_available()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        fused=fused_available
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    print(f"[Training] Starting Cloud Training (Batch: {args.batch_size}, Accum: {args.grad_accum}, Max Steps: {args.max_steps})...")
    model.train()
    step = 0
    running_loss = 0.0
    accum_loss = 0.0
    t0 = time.time()

    for epoch in range(args.epochs):
        optimizer.zero_grad(set_to_none=True)
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=torch.float16):
                _, loss = model(inputs, labels=labels)
                loss_scaled = loss / args.grad_accum

            scaler.scale(loss_scaled).backward()
            accum_loss += loss.item()

            if (batch_idx + 1) % args.grad_accum == 0 or (batch_idx + 1) == len(dataloader):
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                step += 1
                running_loss += accum_loss / args.grad_accum
                accum_loss = 0.0

                if step % 10 == 0 or step == 1 or step >= args.max_steps:
                    elapsed = time.time() - t0
                    cur_loss = running_loss / (10 if step % 10 == 0 and step > 1 else 1)
                    ppl = math.exp(min(cur_loss, 20.0))
                    tokens_per_sec = (step * args.batch_size * args.grad_accum * args.max_seq_len) / max(elapsed, 1e-3)
                    vram_used = torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0
                    print(f"[Step {step:04d}/{args.max_steps}] Loss: {cur_loss:.4f} | PPL: {ppl:.2f} | GradNorm: {grad_norm:.2f} | Speed: {tokens_per_sec:.0f} tok/s | VRAM: {vram_used:.0f} MB")
                    running_loss = 0.0

                if step >= args.max_steps:
                    break

        if step >= args.max_steps:
            break

    print("=" * 78)
    print(f"[Training Complete] Finished {step} steps in {time.time() - t0:.1f}s.")
    print("=" * 78)

    # Export canonical model artifacts for Rust Engine
    export_checkpoint(model, config, tok_path, args.output_dir)

    # Sync to Google Drive if active
    gdrive_dir = os.environ.get("GDRIVE_CHECKPOINT_DIR", "/content/drive/MyDrive/TARA_CHECKPOINTS")
    if os.path.exists("/content/drive/MyDrive"):
        target_gdrive = os.path.join(gdrive_dir, os.path.basename(args.output_dir))
        export_checkpoint(model, config, tok_path, target_gdrive)
        print(f"[GDrive] Checkpoint mirrored to {target_gdrive}")

    print("[Success] Model weights ready for 100% native Rust inference & serving!")


if __name__ == "__main__":
    main()
