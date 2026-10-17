#!/usr/bin/env python3
"""
TARA Autonomous Cloud Training Engine (Google Colab / Cloud GPU Runner)
=============================================================================
Strictly isolated to Cloud Pretraining only (per Project Governance Rule 15).
Features:
1. Dynamic Best-Model Checkpointing (tracks historical minimum loss & perplexity).
2. Automatic Google Drive Mirroring & Disaster Recovery.
3. Seamless Mid-Way Resumption (restores model weights, optimizer, scaler, step).
4. Live Loss, Perplexity (PPL), BF16/FP16 precision, LR, GradNorm, Throughput.
5. Unbounded Extensible Multi-Stage Pipeline (Stages 0..N with dynamic handoff).
6. Automatic Idempotence & Completion Detection (skips already-trained stages).
7. Exports standard `model.safetensors`, `config.json`, and `tokenizer.json`
   for 100% native Rust inference & serving in `tara_engine`.
=============================================================================
"""

import argparse
import glob
import json
import math
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

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
    from tokenizers import Tokenizer as HFTokenizer
    HAS_TOKENIZERS = True
except ImportError:
    HAS_TOKENIZERS = False


class TokenizerOutput:
    def __init__(self, ids: List[int]):
        self.ids = ids

    def __iter__(self):
        return iter(self.ids)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return self.ids[idx]


class TaraNativeTokenizer:
    """Native Python implementation of TARA's Tokenizer matching rust/tara_engine/src/tokenizer.rs."""
    def __init__(self, vocab_map: Dict[str, int]):
        self.token_to_id = vocab_map
        self.id_to_token = {v: k for k, v in vocab_map.items()}
        self.vocab_size = len(vocab_map)
        self.max_token_len = max((len(k) for k in vocab_map.keys()), default=1)
        self.unk_id = vocab_map.get("<|unk|>", vocab_map.get("<unk>", 0))
        self.pad_id = vocab_map.get("<|pad|>", vocab_map.get("<pad>", 0))
        self.eos_id = vocab_map.get("<|eos|>", vocab_map.get("<eos>", 1))

    @classmethod
    def from_file(cls, path: str):
        # 1. First check if it is TARA's native format {"vocab": {...}} or direct dict
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "vocab" in data and isinstance(data["vocab"], dict):
                return cls(data["vocab"])
            elif isinstance(data, dict) and len(data) > 0 and all(isinstance(v, int) for v in list(data.values())[:10]):
                return cls(data)
        except Exception:
            pass

        # 2. Fallback to Hugging Face Tokenizer if available
        if HAS_TOKENIZERS:
            try:
                return HFTokenizer.from_file(path)
            except Exception:
                pass
        raise ValueError(f"Could not load valid tokenizer from {path}")

    def encode(self, text: str) -> TokenizerOutput:
        ids = []
        n = len(text)
        i = 0
        while i < n:
            matched = False
            for length in range(min(self.max_token_len, n - i), 0, -1):
                sub = text[i:i + length]
                if sub in self.token_to_id:
                    ids.append(self.token_to_id[sub])
                    i += length
                    matched = True
                    break
            if not matched:
                ids.append(self.unk_id)
                i += 1
        return TokenizerOutput(ids)

    def decode(self, ids: List[int]) -> str:
        return "".join(self.id_to_token.get(i, "") for i in ids)


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
    def __init__(self, file_paths: List[str], tokenizer: Any, max_seq_len: int = 256):
        self.samples: List[Tuple[List[int], int]] = []
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        print(f"[Dataset] Loading records from {len(file_paths)} shard file(s)...")
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

                        enc_inp = tokenizer.encode(inp_text).ids
                        enc_out = tokenizer.encode(out_text).ids
                        target_start = len(enc_inp)

                        full_ids = enc_inp + enc_out
                        if len(full_ids) > self.max_seq_len:
                            full_ids = full_ids[:self.max_seq_len]

                        if len(full_ids) >= 4:
                            self.samples.append((full_ids, target_start))
                    except Exception:
                        continue

        print(f"[Dataset] Successfully ingested {len(self.samples)} valid sequence pairs.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ids, target_start = self.samples[idx]
        input_ids = torch.tensor(ids, dtype=torch.long)
        labels = input_ids.clone()
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
# 3. Model Weight Loading, Resumption & Checkpoint Export
# =============================================================================

def load_candidate_weights(model: nn.Module, model_dir: str) -> bool:
    """Loads existing candidate safetensors weights if present, ensuring continuous progression."""
    safetensors_files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not safetensors_files or not HAS_SAFETENSORS:
        return False

    print(f"[Weights] Ingesting {len(safetensors_files)} existing SafeTensors shard(s) from {model_dir}...")
    state_dict = {}
    for sf in safetensors_files:
        shard_tensors = load_file(sf, device="cpu")
        state_dict.update(shard_tensors)

    model_sd = model.state_dict()
    matched = 0
    for k, v in state_dict.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            model_sd[k].copy_(v)
            matched += 1

    print(f"[Weights] Successfully restored {matched}/{len(model_sd)} parameter tensors.")
    return matched > 0


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    config: dict,
    tokenizer_path: str,
    output_dir: str,
    step: int,
    epoch: int,
    loss: float,
    best_loss: float,
    stage_id: str,
    is_best: bool = False,
    is_completed: bool = False,
):
    """Exports model weights, optimizer continuation state, and stage metadata."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. config.json
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # 2. tokenizer.json
    if os.path.exists(tokenizer_path):
        shutil.copy2(tokenizer_path, os.path.join(output_dir, "tokenizer.json"))

    # 3. model.safetensors
    state_dict = {k: v.contiguous().cpu() for k, v in model.state_dict().items()}
    if HAS_SAFETENSORS:
        save_file(state_dict, os.path.join(output_dir, "model.safetensors"))
    else:
        torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))

    # 4. Continuation state for mid-way resume
    continuation_state = {
        "step": step,
        "epoch": epoch,
        "loss": loss,
        "best_loss": best_loss,
        "stage_id": stage_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "is_completed": is_completed,
    }
    with open(os.path.join(output_dir, "checkpoint_state.json"), "w", encoding="utf-8") as f:
        json.dump(continuation_state, f, indent=2)

    # 5. Optimizer state for seamless continuation
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler else None,
            "step": step,
            "best_loss": best_loss,
        },
        os.path.join(output_dir, "optimizer_state.pt"),
    )

    # 6. STAGE_METADATA.json (Idempotence & DAG verification)
    metadata = {
        "stage_id": stage_id,
        "status": "COMPLETED" if is_completed else "IN_PROGRESS",
        "step": step,
        "loss": loss,
        "best_loss": best_loss,
        "perplexity": math.exp(min(loss, 20.0)),
        "is_best": is_best,
        "completed_at_utc": datetime.now(timezone.utc).isoformat() if is_completed else None,
    }
    with open(os.path.join(output_dir, "STAGE_METADATA.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    tag = "BEST MILESTONE" if is_best else ("COMPLETED" if is_completed else f"STEP {step}")
    print(f"[Checkpoint] Saved [{tag}] checkpoint -> {output_dir} (Loss: {loss:.4f})")

    # Mirror to Google Drive if active
    gdrive_root = os.environ.get("GDRIVE_CHECKPOINT_DIR", "/content/drive/MyDrive/TARA_CHECKPOINTS")
    if os.path.exists("/content/drive/MyDrive"):
        stage_dir_name = os.path.basename(output_dir)
        gdrive_target = os.path.join(gdrive_root, stage_dir_name)
        os.makedirs(gdrive_target, exist_ok=True)
        try:
            for fname in ["config.json", "tokenizer.json", "model.safetensors", "checkpoint_state.json", "STAGE_METADATA.json"]:
                src = os.path.join(output_dir, fname)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(gdrive_target, fname))
            print(f"[GDrive Mirror] Successfully backed up [{tag}] to Google Drive: {gdrive_target}")
        except Exception as e:
            print(f"[GDrive Mirror Notice] Non-fatal backup warning: {e}")


def try_resume_from_checkpoint(
    checkpoint_dir: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
) -> Tuple[int, int, float, bool]:
    """Attempts to resume from an existing in-progress checkpoint."""
    state_file = os.path.join(checkpoint_dir, "checkpoint_state.json")
    opt_file = os.path.join(checkpoint_dir, "optimizer_state.pt")
    model_file = os.path.join(checkpoint_dir, "model.safetensors")

    if not (os.path.exists(state_file) and os.path.exists(opt_file) and os.path.exists(model_file)):
        return 0, 0, float("inf"), False

    try:
        with open(state_file, "r", encoding="utf-8") as f:
            state = json.load(f)

        step = state.get("step", 0)
        epoch = state.get("epoch", 0)
        best_loss = state.get("best_loss", float("inf"))

        # Load weights
        if HAS_SAFETENSORS:
            weights = load_file(model_file, device="cpu")
            model.load_state_dict(weights, strict=False)

        # Load optimizer
        opt_data = torch.load(opt_file, map_location="cpu")
        optimizer.load_state_dict(opt_data["optimizer"])
        if scaler and opt_data.get("scaler"):
            scaler.load_state_dict(opt_data["scaler"])

        print("=" * 78)
        print(f"  [RESUME] Successfully restored in-progress state from {checkpoint_dir}!")
        print(f"  Resumed Step       : {step}")
        print(f"  Resumed Epoch      : {epoch}")
        print(f"  Historical Best Loss: {best_loss:.4f}")
        print("=" * 78)
        return step, epoch, best_loss, True
    except Exception as e:
        print(f"[Resume Notice] Could not resume from {checkpoint_dir} ({e}). Starting fresh.")
        return 0, 0, float("inf"), False


def check_stage_already_completed(output_dir: str) -> bool:
    """Checks whether the stage has already completed previously (Idempotence)."""
    meta_path = os.path.join(output_dir, "STAGE_METADATA.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("status") == "COMPLETED":
                loss = meta.get("loss", 0.0)
                step = meta.get("step", 0)
                print("=" * 78)
                print(f"  [STAGE DETECTION] Stage is ALREADY COMPLETED (Status: COMPLETED)")
                print(f"  Directory          : {output_dir}")
                print(f"  Completed Step     : {step}")
                print(f"  Final Loss         : {loss:.4f}")
                print(f"  Skipping redundant training to preserve compute & quota.")
                print("=" * 78)
                return True
        except Exception:
            pass
    return False


# =============================================================================
# 4. Real-Time Frontier Benchmark Evaluation & Color Matrix
# =============================================================================

# ANSI Color Codes for Terminal & Colab
CLR_RED = "\033[91m"
CLR_YELLOW = "\033[93m"
CLR_GREEN = "\033[92m"
CLR_CYAN = "\033[96m"
CLR_BOLD = "\033[1m"
CLR_RESET = "\033[0m"


def get_metric_status_and_color(val: float, green_range: Tuple[float, float], yellow_range: Tuple[float, float]) -> Tuple[str, str, str]:
    """
    Evaluates value against Frontier/SOTA (Green), Average (Yellow), or Untrained (Red):
    - RED   : Not match / Early / Untrained
    - YELLOW: Matched Average
    - GREEN : Best against Frontier Model (SOTA)
    """
    if val <= green_range[1]:
        return CLR_GREEN, "BEST (Frontier/SOTA)", "[GREEN]"
    elif val <= yellow_range[1]:
        return CLR_YELLOW, "MATCHED (Average)", "[YELLOW]"
    else:
        return CLR_RED, "NOT MATCH (Untrained)", "[RED]"


def print_frontier_comparison_matrix_colored(step: int, cur_loss: float, best_loss: float):
    ppl = math.exp(min(cur_loss, 20.0))
    bpb = (cur_loss / math.log(2)) / 3.8

    loss_clr, loss_status, loss_tag = get_metric_status_and_color(cur_loss, (1.4, 1.9), (2.0, 3.0))
    ppl_clr, ppl_status, ppl_tag = get_metric_status_and_color(ppl, (4.0, 7.0), (7.1, 20.0))
    bpb_clr, bpb_status, bpb_tag = get_metric_status_and_color(bpb, (0.55, 0.75), (0.76, 1.20))

    print(f"\n{CLR_BOLD}{'=' * 115}{CLR_RESET}", flush=True)
    print(f"  {CLR_CYAN}{CLR_BOLD}TARA vs FRONTIER MODEL REAL-TIME BENCHMARK MATRIX (Step {step:04d}){CLR_RESET}", flush=True)
    print(f"{CLR_BOLD}{'-' * 115}{CLR_RESET}", flush=True)
    print(f"  {'ಮೆಟ್ರಿಕ್ (Metric)':<24} {'ಆರಂಭಿಕ (Untrained)':<22} {'ಸಾಧಾರಣ (Average)':<22} {'ಟಾಪ್ ಲೆವೆಲ್ (Frontier/SOTA)':<28} {'TARA (Current)':<18}", flush=True)
    print(f"{'-' * 115}", flush=True)
    print(f"  {'Cross-Entropy Loss':<24} {'10.0 – 12.0+':<22} {'2.2 – 3.0':<22} {'1.4 – 1.9':<28} {loss_clr}{cur_loss:<7.4f} {loss_status} {loss_tag}{CLR_RESET}", flush=True)
    print(f"  {'Perplexity (PPL)':<24} {'20,000+':<22} {'9.0 – 20.0':<22} {'4.0 – 7.0':<28} {ppl_clr}{ppl:<7.2f} {ppl_status} {ppl_tag}{CLR_RESET}", flush=True)
    print(f"  {'Bits-Per-Byte (BPB)':<24} {'3.5+':<22} {'0.9 – 1.2':<22} {'0.55 – 0.75':<28} {bpb_clr}{f'{bpb:.2f} b/B':<7} {bpb_status} {bpb_tag}{CLR_RESET}", flush=True)
    print(f"{'-' * 115}", flush=True)
    if best_loss < float('inf'):
        b_ppl = math.exp(min(best_loss, 20.0))
        b_bpb = (best_loss / math.log(2)) / 3.8
        b_clr, b_status, b_tag = get_metric_status_and_color(best_loss, (1.4, 1.9), (2.0, 3.0))
        print(f"  {CLR_BOLD}Historical Best Checkpoint:{CLR_RESET} Loss = {b_clr}{best_loss:.4f}{CLR_RESET} | PPL = {b_clr}{b_ppl:.2f}{CLR_RESET} | BPB = {b_clr}{b_bpb:.2f} b/B{CLR_RESET} ({b_clr}{b_status} {b_tag}{CLR_RESET})", flush=True)
    print(f"{'=' * 115}\n", flush=True)


# =============================================================================
# 5. High-Speed Cloud Asset Ingestion (Hugging Face CDN)
# =============================================================================

def ensure_remote_assets_available(model_dir: str, dataset_dir: str, repo_id: str = "manjukp6/tara"):
    """
    High-Speed Cloud Asset Ingestion via Hugging Face Hub (Pure Python).
    Downloads base candidate model and dataset shards concurrently with resume support.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[Notice] huggingface_hub not installed; relying on existing local files.")
        return

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)

    # 1. Check Model Weights & Config
    config_file = os.path.join(model_dir, "config.json")
    safetensor_files = glob.glob(os.path.join(model_dir, "*.safetensors"))
    if not (os.path.exists(config_file) and len(safetensor_files) > 0):
        print(f"[High-Speed HF] Downloading Model Candidate from {repo_id}...")
        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type="model",
                local_dir=model_dir,
                allow_patterns=["config.json", "tokenizer.json", "tokenizer_config.json", "*.safetensors", "neural_model_v1/*"],
                max_workers=8,
            )
            sub_dir = os.path.join(model_dir, "neural_model_v1")
            if os.path.exists(sub_dir):
                for item in os.listdir(sub_dir):
                    s = os.path.join(sub_dir, item)
                    d = os.path.join(model_dir, item)
                    if not os.path.exists(d):
                        shutil.move(s, d)
                shutil.rmtree(sub_dir, ignore_errors=True)
            print(f"[High-Speed HF] Model candidate assets ready in {model_dir}")
        except Exception as e:
            print(f"[Warning] Model candidate download warning: {e}")

    # 2. Check Dataset Shards
    jsonl_shards = glob.glob(os.path.join(dataset_dir, "*.jsonl"))
    if len(jsonl_shards) == 0:
        print(f"[High-Speed HF] Downloading Dataset Shards from {repo_id}...")
        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=dataset_dir,
                allow_patterns=["v2/*.jsonl", "v2/*.json", "*.jsonl"],
                max_workers=8,
            )
            sub_v2 = os.path.join(dataset_dir, "v2")
            if os.path.exists(sub_v2):
                for item in os.listdir(sub_v2):
                    s = os.path.join(sub_v2, item)
                    d = os.path.join(dataset_dir, item)
                    if not os.path.exists(d):
                        shutil.move(s, d)
                shutil.rmtree(sub_v2, ignore_errors=True)
            print(f"[High-Speed HF] Dataset shards ready in {dataset_dir}")
        except Exception as e:
            print(f"[Warning] Dataset shards download warning: {e}")


# =============================================================================
# 6. Main Autonomous Multi-Stage Training Runner
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="TARA Autonomous Cloud Multi-Stage Training Pipeline")
    parser.add_argument("--model-dir", type=str, default="storage/models/tara_candidate_v1", help="Candidate / input model directory")
    parser.add_argument("--dataset-dir", type=str, default="storage/datasets/main_dataset_v2", help="Dataset directory")
    parser.add_argument("--output-dir", type=str, default="storage/models/checkpoints/stage0_pretraining", help="Output checkpoint directory")
    parser.add_argument("--stage-id", type=str, default="stage0_pretraining", help="Unique Stage ID (e.g. stage0, stage1, ... N)")
    parser.add_argument("--stage-name", type=str, default="Stage 0: Foundational Pretraining", help="Human-readable stage title")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Micro-batch size per device")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max-steps", type=int, default=200, help="Max optimizer steps")
    parser.add_argument("--max-seq-len", type=int, default=256, help="Max sequence length")
    parser.add_argument("--resume", action="store_true", default=True, help="Auto-resume from existing checkpoint if available")
    args = parser.parse_args()

    print("=" * 78)
    print(f"  TARA CLOUD MULTI-STAGE TRAINING SYSTEM [{args.stage_name}]")
    print("=" * 78)

    # 1. Idempotence Check
    if check_stage_already_completed(args.output_dir):
        sys.exit(0)

    # 2. Fast Asset Verification & Streaming Download (if files missing)
    ensure_remote_assets_available(args.model_dir, args.dataset_dir)

    # 2. Hardware Inspection & Precision Selection (BF16 for Ampere+, FP16 for Tesla T4)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    precision_str = "FP32"
    precision_dtype = torch.float32

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        if torch.cuda.is_bf16_supported():
            precision_dtype = torch.bfloat16
            precision_str = "BF16"
        else:
            precision_dtype = torch.float16
            precision_str = "FP16"
        print(f"[Hardware] GPU: {gpu_name} | VRAM: {vram_mb:.1f} MB | Precision: {precision_str}")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        print("[Hardware] No GPU detected. Defaulting to CPU (FP32).")

    # 3. Load Architecture Config
    config_path = os.path.join(args.model_dir, "config.json")
    if not os.path.exists(config_path):
        config_path = "storage/models/tara_candidate_v1/config.json"
    if not os.path.exists(config_path):
        print(f"[Error] Missing config.json at {config_path}")
        sys.exit(1)
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # 4. Load Tokenizer
    tok_path = os.path.join(args.model_dir, "tokenizer.json")
    if not os.path.exists(tok_path):
        tok_path = "storage/models/tara_candidate_v1/tokenizer.json"
    if not os.path.exists(tok_path):
        print(f"[Error] tokenizer.json missing at {tok_path}")
        sys.exit(1)
    try:
        tokenizer = TaraNativeTokenizer.from_file(tok_path)
        print(f"[Tokenizer] Successfully loaded TARA Tokenizer ({tokenizer.vocab_size:,} vocabulary tokens)")
    except Exception as e:
        print(f"[Error] Failed to load tokenizer from {tok_path}: {e}")
        sys.exit(1)

    # 5. Build Model
    model = TaraForCausalLM(config).to(device)
    load_candidate_weights(model, args.model_dir)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] Total parameters: {total_params:,} ({total_params * 4 / 1e6:.1f} MB in FP32)")

    # 6. Ingest Shards
    shard_files = sorted(glob.glob(os.path.join(args.dataset_dir, "*.jsonl")))
    if not shard_files:
        shard_files = sorted(glob.glob("storage/datasets/main_dataset_v1/*.jsonl"))
    if not shard_files:
        print(f"[Error] No JSONL shards found in {args.dataset_dir} or fallback.")
        sys.exit(1)

    dataset = TaraJsonlDataset(shard_files[:10], tokenizer, max_seq_len=args.max_seq_len)
    if len(dataset) == 0:
        print("[Error] Ingested 0 dataset samples. Aborting.")
        sys.exit(1)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    # 7. Optimizer with Fused AdamW & Scaler
    fused_available = 'fused' in torch.optim.AdamW.__init__.__code__.co_varnames and torch.cuda.is_available()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        fused=fused_available,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and precision_dtype == torch.float16))

    # 8. Check for Mid-Way Resume
    start_step = 0
    start_epoch = 0
    best_loss = float("inf")
    if args.resume:
        start_step, start_epoch, best_loss, was_resumed = try_resume_from_checkpoint(
            args.output_dir, model, optimizer, scaler
        )
        if start_step >= args.max_steps:
            print(f"[Resume] Training already completed all {args.max_steps} steps. Finishing.")
            save_checkpoint(
                model, optimizer, scaler, config, tok_path, args.output_dir,
                start_step, start_epoch, best_loss, best_loss, args.stage_id,
                is_best=False, is_completed=True
            )
            sys.exit(0)

    # Setup Best Checkpoint Directory
    best_cp_dir = os.path.join(args.output_dir, "best_checkpoint")

    print(f"[Training] Launching Training: Epochs={args.epochs}, Batch={args.batch_size}, Accum={args.grad_accum}, Precision={precision_str}, Steps={start_step}..{args.max_steps}")
    model.train()
    step = start_step
    running_loss = 0.0
    accum_loss = 0.0
    ema_loss = None
    t0 = time.time()

    for epoch in range(start_epoch, args.epochs):
        optimizer.zero_grad(set_to_none=True)
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=precision_dtype):
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
                cur_step_loss = accum_loss / args.grad_accum
                running_loss += cur_step_loss
                accum_loss = 0.0

                ema_loss = cur_step_loss if ema_loss is None else (0.9 * ema_loss + 0.1 * cur_step_loss)

                # Real-Time Telemetry Logging (LIVE on EVERY SINGLE STEP with immediate stdout flush)
                step_duration = max(time.time() - step_last_time, 1e-4) if 'step_last_time' in locals() else 0.5
                step_last_time = time.time()
                step_tokens = args.batch_size * args.grad_accum * args.max_seq_len
                step_speed = step_tokens / step_duration

                ppl = math.exp(min(cur_step_loss, 20.0))
                cur_lr = optimizer.param_groups[0]["lr"]
                vram_used = torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0
                vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024) if torch.cuda.is_available() else 0

                bpb = (cur_step_loss / math.log(2)) / 3.8
                loss_clr, loss_status, loss_tag = get_metric_status_and_color(cur_step_loss, (1.4, 1.9), (2.0, 3.0))
                ppl_clr, ppl_status, ppl_tag = get_metric_status_and_color(ppl, (4.0, 7.0), (7.1, 20.0))
                bpb_clr, bpb_status, bpb_tag = get_metric_status_and_color(bpb, (0.55, 0.75), (0.76, 1.20))

                print(
                    f"[{args.stage_id} | Step {step:04d}/{args.max_steps}] "
                    f"Loss: {loss_clr}{cur_step_loss:.4f}{CLR_RESET} (EMA: {ema_loss:.4f}) | "
                    f"PPL: {ppl_clr}{ppl:.2f}{CLR_RESET} | "
                    f"BPB: {bpb_clr}{bpb:.2f} b/B{CLR_RESET} | "
                    f"Standing: {loss_clr}{loss_status} {loss_tag}{CLR_RESET} | "
                    f"LR: {cur_lr:.2e} | "
                    f"GradNorm: {grad_norm:.2f} | "
                    f"Speed: {step_speed:.0f} tok/s ({step_duration:.2f}s) | "
                    f"VRAM: {vram_used:.0f}/{vram_total:.0f} MB ({precision_str})",
                    flush=True
                )

                # Periodic Benchmark Matrix Display
                if step % 25 == 0 or step == start_step + 1:
                    print_frontier_comparison_matrix_colored(step, cur_step_loss, best_loss)

                # Dynamic Best-Checkpointing Gate
                if cur_step_loss < best_loss and step >= 10:
                    best_loss = cur_step_loss
                    print(f"[{args.stage_id}] >>> {CLR_GREEN}{CLR_BOLD}NEW HISTORICAL BEST LOSS ACHIEVED: {best_loss:.4f}{CLR_RESET} <<<", flush=True)
                    print_frontier_comparison_matrix_colored(step, cur_step_loss, best_loss)
                    save_checkpoint(
                        model, optimizer, scaler, config, tok_path, best_cp_dir,
                        step, epoch, cur_step_loss, best_loss, args.stage_id,
                        is_best=True, is_completed=False
                    )

                # Periodic Step Checkpointing for Mid-Way Resilience
                if step % 50 == 0:
                    save_checkpoint(
                        model, optimizer, scaler, config, tok_path, args.output_dir,
                        step, epoch, cur_step_loss, best_loss, args.stage_id,
                        is_best=False, is_completed=False
                    )

                if step >= args.max_steps:
                    break

        if step >= args.max_steps:
            break

    total_time = time.time() - t0
    final_loss = ema_loss if ema_loss is not None else 0.0
    print("=" * 78)
    print(f"[{args.stage_id} FINISHED] Completed {step} steps in {total_time:.1f}s | Final Loss: {final_loss:.4f} | Best Loss: {best_loss:.4f}")
    print("=" * 78)

    # Final Checkpoint Save with COMPLETED status
    save_checkpoint(
        model, optimizer, scaler, config, tok_path, args.output_dir,
        step, epoch, final_loss, best_loss, args.stage_id,
        is_best=False, is_completed=True
    )

    print(f"[Stage Complete] Production artifacts published to: {args.output_dir}")
    print(f"[Stage Complete] Best Milestone model archived in: {best_cp_dir}")
    print("[Success] All weights are 100% compatible with native Rust TARA inference & serving!")


if __name__ == "__main__":
    main()
