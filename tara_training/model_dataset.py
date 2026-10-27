#!/usr/bin/env python3
"""
TARA Autonomous Cloud Training - Model Architecture and Streaming Dataset Engine
================================================================================
Handles:
1. Native PyTorch Tara Causal Architecture (TaraForCausalLM, TaraAttention, RMSNorm,
   SwiGLU TaraMlp, RoPE rotation, Multi-Modal Encoders and CausalLMOutput).
2. Native Tokenizer (TaraNativeTokenizer, TokenizerOutput, byte fallback).
3. Streaming JSONL Dataset with JIT shard downloading and verification.
4. Shard Ledger, line indexing, and multi-domain weighted scheduling.
5. Neural World Model state transition datasets and candidate weight loading.
6. Standalone with zero dependency on local PC or external repository code.
================================================================================
"""

import copy
import gc
import glob
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset

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

# Import filesystem and integrity helpers from gdrive_sync
from gdrive_sync import _safe_unlink, _safe_rmtree, compute_file_sha256, _stream_copy_file

class TokenizerOutput:
    def __init__(self, ids: List[int]):
        self.ids = ids

    def __iter__(self):
        return iter(self.ids)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return self.ids[idx]


SPECIAL_TOKENS = [
    "<|pad|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|unk|>",
    "<|creator_auth|>",
    "<|tara_rule|>",
    "<|tara_exec|>",
    "<|tara_skill|>",
    "<|tara_memory|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|audio_start|>",
    "<|audio_end|>",
    "[STATE]",
    "[ACTION]",
    "[NEXT_STATE]",
    "[EOS]",
]


class TaraNativeTokenizer:
    """Native Python implementation of TARA's Tokenizer matching rust/tara_engine/src/tokenizer.rs."""
    def __init__(self, vocab_map: Dict[str, int]):
        self.token_to_id = vocab_map
        self.id_to_token = {v: k for k, v in vocab_map.items()}
        self.vocab_size = len(vocab_map)
        self.max_token_len = max((len(k) for k in vocab_map.keys()), default=1)
        self.unk_id = vocab_map.get("<|unk|>", vocab_map.get("<unk>", 0))
        self.pad_id = vocab_map.get("<|pad|>", vocab_map.get("<pad>", 0))
        self.eos_id = vocab_map.get("<|eos|>", vocab_map.get("<|im_end|>", vocab_map.get("<eos>", 1)))
        self.im_start_id = vocab_map.get("<|im_start|>", 1)
        self.im_end_id = vocab_map.get("<|im_end|>", 2)

        # Pre-populate fast 256 byte lookup tables for <0x00> through <0xFF>
        self.byte_to_id: List[Optional[int]] = [None] * 256
        self.id_to_byte: Dict[int, int] = {}
        for b in range(256):
            byte_token_upper = f"<0x{b:02X}>"
            byte_token_lower = f"<0x{b:02x}>"
            tok_token = f"<tok_{b}>"
            if byte_token_upper in self.token_to_id:
                bid = self.token_to_id[byte_token_upper]
                self.byte_to_id[b] = bid
                self.id_to_byte[bid] = b
            elif byte_token_lower in self.token_to_id:
                bid = self.token_to_id[byte_token_lower]
                self.byte_to_id[b] = bid
                self.id_to_byte[bid] = b
            elif tok_token in self.token_to_id:
                bid = self.token_to_id[tok_token]
                self.byte_to_id[b] = bid
                self.id_to_byte[bid] = b

    def get_vocab_size(self) -> int:
        return self.vocab_size

    @classmethod
    def from_file(cls, path: str):
        # 1. Primary: Use Hugging Face C/Rust tokenizer if file adheres to HF format
        if HAS_TOKENIZERS:
            try:
                tok = HFTokenizer.from_file(path)
                print("[Tokenizer] Ultra-fast native Hugging Face C/Rust tokenizer backend activated!")
                return tok
            except Exception:
                pass

        # 2. Fallback: Parse vocab map directly from TARA native tokenizer JSON format
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "vocab" in data and isinstance(data["vocab"], dict):
                print(f"[Tokenizer] Loaded TARA Native Vocabulary ({len(data['vocab']):,} tokens).")
                return cls(data["vocab"])
            elif isinstance(data, dict) and "model" in data and isinstance(data["model"], dict) and "vocab" in data["model"] and isinstance(data["model"]["vocab"], dict):
                print(f"[Tokenizer] Loaded TARA Model Vocabulary ({len(data['model']['vocab']):,} tokens).")
                return cls(data["model"]["vocab"])
            elif isinstance(data, dict) and len(data) > 0 and all(isinstance(v, int) for v in list(data.values())[:10]):
                print(f"[Tokenizer] Loaded Flat Vocabulary Map ({len(data):,} tokens).")
                return cls(data)
        except Exception as err:
            print(f"[Tokenizer Notice] Native JSON load attempt: {err}")

        raise ValueError(f"Could not load valid tokenizer from {path}")

    def encode(self, text: str) -> TokenizerOutput:
        ids: List[int] = []
        n = len(text)
        i = 0
        while i < n:
            # 1. Special token priority pass
            matched_special = False
            for sp in SPECIAL_TOKENS:
                if text.startswith(sp, i):
                    if sp in self.token_to_id:
                        ids.append(self.token_to_id[sp])
                        i += len(sp)
                        matched_special = True
                        break
            if matched_special:
                continue

            # 2. Greedy longest-match subword search
            matched = False
            max_search_len = min(self.max_token_len, n - i)
            for length in range(max_search_len, 0, -1):
                sub = text[i : i + length]
                if sub in self.token_to_id:
                    tid = self.token_to_id[sub]
                    # Byte fallback tokens (<0x..>) are strictly reserved for fallback
                    if tid not in self.id_to_byte:
                        ids.append(tid)
                        i += length
                        matched = True
                        break

            # 3. Zero unknown byte fallback
            if not matched:
                ch = text[i]
                i += 1
                ch_bytes = ch.encode("utf-8")
                if not self.id_to_byte:
                    ids.append(self.unk_id)
                else:
                    for b in ch_bytes:
                        byte_id = self.byte_to_id[b]
                        if byte_id is not None:
                            ids.append(byte_id)
                        else:
                            ids.append(self.unk_id)

        return TokenizerOutput(ids)

    def decode(self, ids: List[int]) -> str:
        skip_ids = {self.pad_id, self.im_start_id, self.im_end_id}
        byte_buf = bytearray()
        for token_id in ids:
            if token_id in skip_ids:
                continue
            if token_id in self.id_to_byte:
                byte_buf.append(self.id_to_byte[token_id])
            elif token_id in self.id_to_token:
                byte_buf.extend(self.id_to_token[token_id].encode("utf-8"))
        return byte_buf.decode("utf-8", errors="replace")


# =============================================================================
# 1. Native PyTorch Tara Causal Architecture, Multi-Modal Projectors & Tokenizer
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


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half_dim = x.shape[-1] // 2
    return torch.cat((-x[..., half_dim:], x[..., :half_dim]), dim=-1)


def resolve_dynamic_rope_theta(config: Optional[dict] = None) -> float:
    """Dynamically resolve RoPE base frequency from configuration or environment."""
    if config and "rope_theta" in config:
        try:
            val = float(config["rope_theta"])
            if val > 0.0 and math.isfinite(val):
                return val
        except (ValueError, TypeError):
            pass
    env_theta = os.environ.get("TARA_ROPE_THETA", os.environ.get("ROPE_THETA", "")).strip()
    if env_theta:
        try:
            val = float(env_theta)
            if val > 0.0 and math.isfinite(val):
                return val
        except ValueError:
            pass
    return 500000.0


def precompute_rope_cos_sin(dim: int, end: int, theta: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    if theta is None:
        theta = resolve_dynamic_rope_theta()
    half_dim = dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, half_dim, dtype=torch.float32) * 2.0 / dim))
    t = torch.arange(end, dtype=torch.float32)
    angles = torch.outer(t, freqs)  # [end, half_dim]
    cos_half = torch.cos(angles)
    sin_half = torch.sin(angles)
    # Concatenate across both halves to match Rust rope.rs rotate-half layout
    cos = torch.cat([cos_half, cos_half], dim=-1)  # [end, dim]
    sin = torch.cat([sin_half, sin_half], dim=-1)  # [end, dim]
    return cos, sin


def apply_rope(xq: torch.Tensor, xk: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # xq: [batch, seq_len, n_heads, head_dim]
    # xk: [batch, seq_len, n_kv_heads, head_dim]
    # cos, sin: [seq_len, head_dim] -> unsqueeze to [1, seq_len, 1, head_dim]
    seqlen = xq.shape[1]
    c = cos[:seqlen].unsqueeze(0).unsqueeze(2).type_as(xq)
    s = sin[:seqlen].unsqueeze(0).unsqueeze(2).type_as(xq)
    xq_out = (xq * c) + (rotate_half(xq) * s)
    xk_out = (xk * c) + (rotate_half(xk) * s)
    return xq_out, xk_out


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

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        q = self.q_proj(x).view(bsz, seqlen, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        q, k = apply_rope(q, k, cos, sin)

        # GQA repeat KV
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=2)
            v = v.repeat_interleave(self.n_rep, dim=2)

        # Transpose to [bsz, n_heads, seqlen, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Flash / SDPA scaled dot product attention with guaranteed causality
        if mask is not None and not (mask.dim() == 2 and mask.bool().all()):
            if mask.dim() == 2:
                causal_mask = torch.triu(torch.full((seqlen, seqlen), float("-inf"), device=x.device, dtype=x.dtype), diagonal=1)
                pad_mask = torch.zeros((bsz, 1, 1, seqlen), device=x.device, dtype=x.dtype)
                pad_mask = pad_mask.masked_fill(~mask.unsqueeze(1).unsqueeze(2).bool(), float("-inf"))
                combined_mask = causal_mask.unsqueeze(0).unsqueeze(0) + pad_mask
                scores = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, attn_mask=combined_mask, is_causal=False
                )
            else:
                scores = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, attn_mask=mask, is_causal=False
                )
        else:
            scores = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, is_causal=(seqlen > 1)
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

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = x + self.self_attn(self.input_layernorm(x), cos, sin, mask)
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out


class TaraModel(nn.Module):
    def __init__(self, config: dict, gradient_checkpointing: bool = False):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = gradient_checkpointing
        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.layers = nn.ModuleList([
            TaraDecoderLayer(config, i) for i in range(config["num_hidden_layers"])
        ])
        self.norm = RMSNorm(config["hidden_size"], eps=config.get("rms_norm_eps", 1e-5))

        # Precompute RoPE cos/sin tables matching Rust rope.rs rotate-half layout
        head_dim = config.get("head_dim", config["hidden_size"] // config["num_attention_heads"])
        max_pos = config.get("max_position_embeddings", 4096)
        theta = resolve_dynamic_rope_theta(config)
        cos, sin = precompute_rope_cos_sin(head_dim, max_pos, theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if input_ids is not None and input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if inputs_embeds is not None and inputs_embeds.dim() == 2:
            inputs_embeds = inputs_embeds.unsqueeze(0)
        if inputs_embeds is not None:
            h = inputs_embeds
            seqlen = h.shape[1]
        elif input_ids is not None:
            seqlen = input_ids.shape[1]
            h = self.embed_tokens(input_ids)
        else:
            raise ValueError("Either input_ids or inputs_embeds must be provided")

        if seqlen > self.rope_cos.shape[0]:
            head_dim = self.config.get("head_dim", self.config["hidden_size"] // self.config["num_attention_heads"])
            theta = resolve_dynamic_rope_theta(self.config)
            new_max = max(seqlen, self.rope_cos.shape[0] * 2)
            new_cos, new_sin = precompute_rope_cos_sin(head_dim, new_max, theta)
            self.rope_cos = new_cos.to(h.device)
            self.rope_sin = new_sin.to(h.device)

        cos = self.rope_cos[:seqlen]
        sin = self.rope_sin[:seqlen]

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(inputs[0], inputs[1], inputs[2], inputs[3] if len(inputs) > 3 else None)
                    return custom_forward
                h = torch.utils.checkpoint.checkpoint(create_custom_forward(layer), h, cos, sin, attention_mask, use_reentrant=False)
            else:
                h = layer(h, cos, sin, attention_mask)

        return self.norm(h)


class CrossModalProjector(nn.Module):
    """Native cross-modal projector matching rust/tara_engine/src/multimodal/projector.rs."""
    def __init__(self, projector_type: str, input_dim: int, output_dim: int, intermediate_dim: int):
        super().__init__()
        self.projector_type = projector_type.lower()
        if self.projector_type == "mlp":
            self.net = nn.Sequential(
                nn.Linear(input_dim, intermediate_dim),
                nn.GELU(),
                nn.Linear(intermediate_dim, output_dim),
            )
        else:
            self.net = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PyTorchVisionEncoder(nn.Module):
    """Native 2D patch vision encoder matching rust/tara_engine/src/multimodal/vision.rs."""
    def __init__(self, config: dict):
        super().__init__()
        self.patch_height = config.get("patch_height", 16)
        self.patch_width = config.get("patch_width", 16)
        self.channels = config.get("channels", 3)
        self.hidden_size = config.get("hidden_size", 2048)
        self.intermediate_dim = config.get("intermediate_dim", self.hidden_size * 2)
        self.projector_type = config.get("projector_type", "mlp")
        patch_dim = self.patch_height * self.patch_width * self.channels
        self.projector = CrossModalProjector(self.projector_type, patch_dim, self.hidden_size, self.intermediate_dim)
        self.normalize_pixels = config.get("normalize_pixels", True)
        self.register_buffer("mean", torch.tensor(config.get("mean", [0.485, 0.456, 0.406])).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(config.get("std", [0.229, 0.224, 0.225])).view(1, -1, 1, 1))

    def generate_2d_pos_embeddings(self, num_y: int, num_x: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        d_half = self.hidden_size // 2
        d_y = d_half // 2
        d_x = (self.hidden_size - d_half) // 2

        py = torch.arange(1, num_y + 1, device=device, dtype=dtype).unsqueeze(1)
        px = torch.arange(1, num_x + 1, device=device, dtype=dtype).unsqueeze(1)

        div_y = 10000.0 ** (torch.arange(0, d_y, device=device, dtype=dtype) * 2 / d_half)
        div_x = 10000.0 ** (torch.arange(0, d_x, device=device, dtype=dtype) * 2 / (self.hidden_size - d_half))

        angles_y = py / div_y
        angles_x = px / div_x

        y_emb = torch.stack([angles_y.sin(), angles_y.cos()], dim=-1).flatten(1)
        x_emb = torch.stack([angles_x.sin(), angles_x.cos()], dim=-1).flatten(1)

        y_grid = y_emb.unsqueeze(1).expand(-1, num_x, -1)
        x_grid = x_emb.unsqueeze(0).expand(num_y, -1, -1)
        return torch.cat([y_grid, x_grid], dim=-1).reshape(-1, self.hidden_size)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.dim() == 3:
            images = images.unsqueeze(0)
        B, C, H, W = images.shape
        if self.normalize_pixels:
            images = (images - self.mean.to(images.device)) / self.std.to(images.device).clamp(min=1e-5)

        num_y = H // self.patch_height
        num_x = W // self.patch_width
        patches = images.unfold(2, self.patch_height, self.patch_height).unfold(3, self.patch_width, self.patch_width)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous().reshape(B, num_y * num_x, -1)
        projected = self.projector(patches)
        pos = self.generate_2d_pos_embeddings(num_y, num_x, images.device, projected.dtype)
        return projected + pos.unsqueeze(0)


class PyTorchAudioEncoder(nn.Module):
    """Native audio Mel-spectrogram encoder matching rust/tara_engine/src/multimodal/audio.rs."""
    def __init__(self, config: dict):
        super().__init__()
        self.sample_rate = config.get("sample_rate", 16000)
        self.n_fft = config.get("n_fft", 512)
        self.hop_length = config.get("hop_length", 160)
        self.n_mels = config.get("n_mels", 80)
        self.hidden_size = config.get("hidden_size", 2048)
        self.intermediate_dim = config.get("intermediate_dim", self.hidden_size * 2)
        self.projector_type = config.get("projector_type", "mlp")
        self.projector = CrossModalProjector(self.projector_type, self.n_mels, self.hidden_size, self.intermediate_dim)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        if audio.dim() == 3:
            audio = audio.squeeze(1)
        window = torch.hann_window(self.n_fft, device=audio.device)
        stft = torch.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length, window=window, return_complex=True)
        mag = torch.abs(stft).transpose(1, 2)
        if mag.shape[-1] != self.n_mels:
            mag = torch.nn.functional.adaptive_avg_pool1d(mag, self.n_mels)
        log_mel = torch.log(mag.clamp(min=1e-5))
        return self.projector(log_mel)


class CausalLMOutput(tuple):
    """
    Output of TaraForCausalLM.forward that functions simultaneously as:
    1. A 4-tuple: (logits, total_loss, ce_loss, aux_loss) for direct tuple unpacking:
       logits, loss, ce_loss, aux_loss = model(...)
    2. An object with .loss, .logits, .ce_loss, .aux_loss attribute access.
    3. A dict-like container with ["loss"], ["logits"], etc.
    """
    def __new__(cls, logits, total_loss=None, ce_loss=None, aux_loss=None):
        return super().__new__(cls, (logits, total_loss, ce_loss, aux_loss))

    @property
    def logits(self): return self[0]
    @property
    def loss(self): return self[1] if self[1] is not None else self[2]
    @property
    def ce_loss(self): return self[2]
    @property
    def aux_loss(self): return self[3]
    @property
    def total_loss(self): return self[1]

    def __getitem__(self, key):
        if isinstance(key, str):
            mapping = {
                "logits": self[0],
                "loss": self[1] if self[1] is not None else self[2],
                "total_loss": self[1],
                "ce_loss": self[2],
                "aux_loss": self[3],
            }
            if key in mapping:
                return mapping[key]
            raise KeyError(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


class TaraForCausalLM(nn.Module):
    def __init__(self, config: dict, gradient_checkpointing: bool = False):
        super().__init__()
        self.config = config
        self.model = TaraModel(config, gradient_checkpointing=gradient_checkpointing)
        self.lm_head = nn.Linear(config["hidden_size"], config["vocab_size"], bias=False)

        # Tie weights if requested
        if config.get("tie_word_embeddings", False):
            self.lm_head.weight = self.model.embed_tokens.weight

        # MultiModal perception encoders (enabled when multimodal is active)
        self.vision_encoder = None
        self.audio_encoder = None
        if config.get("multimodal", False) or "vision_config" in config:
            v_cfg = config.get("vision_config", {"hidden_size": config["hidden_size"]})
            self.vision_encoder = PyTorchVisionEncoder(v_cfg)
        if config.get("multimodal", False) or "audio_config" in config:
            a_cfg = config.get("audio_config", {"hidden_size": config["hidden_size"]})
            self.audio_encoder = PyTorchAudioEncoder(a_cfg)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if input_ids is not None and input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if labels is not None and labels.dim() == 1:
            labels = labels.unsqueeze(0)
        if inputs_embeds is not None and inputs_embeds.dim() == 2:
            inputs_embeds = inputs_embeds.unsqueeze(0)
        if inputs_embeds is None:
            if input_ids is not None:
                embeds = self.model.embed_tokens(input_ids)
            else:
                raise ValueError("Either input_ids or inputs_embeds must be provided")

            if images is not None:
                if self.vision_encoder is None:
                    v_cfg = self.config.get("vision_config", {"hidden_size": self.config["hidden_size"]})
                    self.vision_encoder = PyTorchVisionEncoder(v_cfg).to(embeds.device)
                v_patches = self.vision_encoder(images)
                v_start_id = self.config.get("vision_start_token_id", 9)
                v_end_id = self.config.get("vision_end_token_id", 10)
                v_start = self.model.embed_tokens(torch.tensor([[v_start_id]], device=embeds.device)).expand(v_patches.size(0), -1, -1)
                v_end = self.model.embed_tokens(torch.tensor([[v_end_id]], device=embeds.device)).expand(v_patches.size(0), -1, -1)
                v_seq = torch.cat([v_start, v_patches, v_end], dim=1)
                embeds = torch.cat([v_seq, embeds], dim=1)
                if labels is not None:
                    img_mask = torch.full((labels.size(0), v_seq.size(1)), -100, dtype=labels.dtype, device=labels.device)
                    labels = torch.cat([img_mask, labels], dim=1)

            if audio is not None:
                if self.audio_encoder is None:
                    a_cfg = self.config.get("audio_config", {"hidden_size": self.config["hidden_size"]})
                    self.audio_encoder = PyTorchAudioEncoder(a_cfg).to(embeds.device)
                a_frames = self.audio_encoder(audio)
                a_start_id = self.config.get("audio_start_token_id", 11)
                a_end_id = self.config.get("audio_end_token_id", 12)
                a_start = self.model.embed_tokens(torch.tensor([[a_start_id]], device=embeds.device)).expand(a_frames.size(0), -1, -1)
                a_end = self.model.embed_tokens(torch.tensor([[a_end_id]], device=embeds.device)).expand(a_frames.size(0), -1, -1)
                a_seq = torch.cat([a_start, a_frames, a_end], dim=1)
                embeds = torch.cat([a_seq, embeds], dim=1)
                if labels is not None:
                    aud_mask = torch.full((labels.size(0), a_seq.size(1)), -100, dtype=labels.dtype, device=labels.device)
                    labels = torch.cat([aud_mask, labels], dim=1)

            inputs_embeds = embeds

        hidden_states = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)

        total_loss = None
        ce_loss = None
        aux_loss = None
        logits = None
        if labels is not None:
            # Shift for causal next-token prediction
            shift_hidden = hidden_states[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Memory-Efficient Chunked Cross-Entropy Loss to eliminate CUDA OOM
            # Instead of materializing full [B, S, V] logits (over 6.5 GB for 100k vocab),
            # compute CE loss in small token chunks.
            flat_hidden = shift_hidden.view(-1, shift_hidden.size(-1))
            flat_labels = shift_labels.view(-1)
            valid_mask = (flat_labels != -100)
            total_valid = valid_mask.sum()

            if total_valid > 0:
                loss_sum = torch.tensor(0.0, device=flat_hidden.device, dtype=torch.float32)
                # Chunk size 1024 eliminates kernel launch storm (4x fewer launches) while keeping logits VRAM safely bounded (~300MB)
                chunk_sz = 1024
                for chunk_start in range(0, flat_hidden.size(0), chunk_sz):
                    chunk_end = min(chunk_start + chunk_sz, flat_hidden.size(0))
                    c_mask = valid_mask[chunk_start:chunk_end]
                    if not c_mask.any():
                        continue
                    c_h = flat_hidden[chunk_start:chunk_end][c_mask]
                    c_l = flat_labels[chunk_start:chunk_end][c_mask]
                    c_logits = self.lm_head(c_h)
                    c_loss = F.cross_entropy(c_logits.float(), c_l, reduction="sum")
                    loss_sum = loss_sum + c_loss
                ce_loss = loss_sum / total_valid.clamp(min=1)
            else:
                ce_loss = torch.tensor(0.0, device=flat_hidden.device, requires_grad=True)

            # Auxiliary representation health regularizer: penalize representation collapse on valid tokens only
            if total_valid > 0:
                valid_hidden = flat_hidden[valid_mask]
                hidden_var = valid_hidden.var(dim=-1).mean()
                aux_loss = 0.005 * torch.relu(1.0 - hidden_var)
            else:
                aux_loss = torch.tensor(0.0, device=flat_hidden.device)
            total_loss = ce_loss + aux_loss

            # In training mode with large batches, omit full logits to save 3+ GB VRAM.
            # In eval mode or small test runs (<= 128 tokens), compute full logits.
            if not self.training or flat_hidden.size(0) <= 128:
                logits = self.lm_head(hidden_states)
            else:
                logits = None
        else:
            logits = self.lm_head(hidden_states)

        return CausalLMOutput(logits, total_loss, ce_loss, aux_loss)


class TaraMultiModalForCausalLM(TaraForCausalLM):
    """Explicit MultiModal architecture binding vision and audio encoders."""
    def __init__(self, config: dict, gradient_checkpointing: bool = False):
        mm_config = dict(config)
        mm_config["multimodal"] = True
        super().__init__(mm_config, gradient_checkpointing=gradient_checkpointing)


# Dedicated Neural World Model Architecture (matches storage/models/tara_world_model)
TaraWorldModel = TaraForCausalLM


# =============================================================================
# 2. High-Throughput Streaming Dataset, Persistent Shard Ledger & JIT Downloader
# =============================================================================

_token_logged: bool = False

def _resolve_hf_token(cli_token: Optional[str] = None) -> Optional[str]:
    """
    Securely resolves the Hugging Face access token across multiple environments:
    1. Direct CLI argument (--hf-token).
    2. Direct HF_TOKEN environment variable.
    3. Google Colab userdata secret 'tara_sync' or 'HF_TOKEN'.
    4. Hugging Face Hub stored cache token (get_token()).
    """
    global _token_logged
    resolved: Optional[str] = None

    if cli_token and cli_token.strip():
        resolved = cli_token.strip()
        os.environ["HF_TOKEN"] = resolved
    elif os.environ.get("HF_TOKEN") and os.environ.get("HF_TOKEN").strip():
        resolved = os.environ.get("HF_TOKEN").strip()
    else:
        # Check Google Colab userdata secrets if running in Colab
        try:
            from google.colab import userdata
            for secret_key in ("tara_sync", "HF_TOKEN"):
                try:
                    colab_token = userdata.get(secret_key)
                    if colab_token and colab_token.strip():
                        resolved = colab_token.strip()
                        os.environ["HF_TOKEN"] = resolved
                        break
                except Exception:
                    pass
        except Exception:
            pass

        if not resolved:
            try:
                from huggingface_hub import get_token
                cached = get_token()
                if cached and cached.strip():
                    resolved = cached.strip()
                    os.environ["HF_TOKEN"] = resolved
            except Exception:
                pass

    if not _token_logged:
        _token_logged = True
        if resolved:
            print(f"[Auth] Hugging Face authentication token resolved (length: {len(resolved)}).", flush=True)
        else:
            print("[Auth Notice] No Hugging Face token resolved; unauthenticated access to public resources only.", flush=True)

    return resolved


def download_single_shard_jit(
    repo_id: str,
    prefix: str,
    shard_name: str,
    local_dir: str,
    max_shard_bytes: int = 0,
) -> str:
    """
    Downloads ONLY a single JSONL shard on-demand from Hugging Face.

    Maintains strictly a single active shard on disk, eliminating bulk downloads.
    An enforced maximum shard size can be specified via ``max_shard_bytes``
    (default 0 = unlimited).  When the downloaded file exceeds the limit the
    incomplete temporary file is removed and a ``RuntimeError`` is raised —
    original dataset files and user-owned files are never deleted.
    """
    target_path = os.path.join(local_dir, shard_name)
    target_dir = os.path.dirname(target_path)
    os.makedirs(target_dir if target_dir else local_dir, exist_ok=True)
    if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
        if max_shard_bytes > 0:
            existing_size = os.path.getsize(target_path)
            if existing_size > max_shard_bytes:
                raise RuntimeError(
                    f"[JIT Downloader] Cached shard '{shard_name}' ({existing_size:,} bytes) exceeds "
                    f"enforced maximum shard size ({max_shard_bytes:,} bytes). "
                    f"Remove the oversized file from '{target_path}' and adjust --max-shard-bytes."
                )
        return target_path

    token = _resolve_hf_token()
    remote_path = f"{prefix}/{shard_name}" if (prefix and not shard_name.startswith(f"{prefix}/")) else shard_name
    print(f"[JIT Downloader] Fetching single shard '{remote_path}' from https://huggingface.co/{repo_id}...", flush=True)

    max_retries = 5
    tmp_path: Optional[str] = None
    for attempt in range(1, max_retries + 1):
        try:
            from huggingface_hub import hf_hub_download
            dl_path = hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=remote_path,
                local_dir=local_dir,
                token=token,
            )
            tmp_path = dl_path if dl_path != target_path else None

            # Enforce maximum shard size BEFORE promoting to the target path.
            # Only TARA-owned temporary download files are removed on violation;
            # original dataset files and user files are never deleted.
            if max_shard_bytes > 0 and os.path.exists(dl_path):
                dl_size = os.path.getsize(dl_path)
                if dl_size > max_shard_bytes:
                    if tmp_path and os.path.exists(tmp_path) and os.path.abspath(tmp_path) != os.path.abspath(target_path):
                        try:
                            os.remove(tmp_path)
                        except Exception:
                            pass
                    raise RuntimeError(
                        f"[JIT Downloader] Shard '{shard_name}' ({dl_size:,} bytes) exceeds enforced "
                        f"maximum shard size ({max_shard_bytes:,} bytes). "
                        f"Download aborted; incomplete temporary file removed. "
                        f"Adjust --max-shard-bytes or choose a smaller shard."
                    )

            if dl_path != target_path and os.path.exists(dl_path):
                shutil.move(dl_path, target_path)
                sub_dir = os.path.dirname(dl_path)
                if sub_dir != local_dir and os.path.exists(sub_dir) and not os.listdir(sub_dir):
                    try:
                        os.rmdir(sub_dir)
                    except Exception:
                        pass
            if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
                print(f"[JIT Downloader] Shard '{shard_name}' verified ({os.path.getsize(target_path)/(1024*1024):.1f} MB).", flush=True)
                return target_path
        except RuntimeError:
            raise  # Re-raise size-bound violations without retry
        except Exception as err:
            print(f"[JIT Downloader Warning] Attempt {attempt}/{max_retries} failed to download '{shard_name}': {err}", flush=True)
            time.sleep(attempt * 4)

    raise RuntimeError(f"[JIT Downloader Fatal] Failed to download {shard_name} after {max_retries} attempts.")


class ShardLedger:
    """
    Authoritative Persistent Incremental Stage Ledger.

    Tracks completed shards, runtime dynamic SHA-256 digests, and active
    intra-shard progress across an unlimited number of incremental training
    stages. Persisted to local checkpoint directory and mirrored continuously
    locally.

    Schema version 3.0 adds explicit stage identity and per-shard training-unit
    accounting required for Stage 0 → Stage 1 → Stage N incremental workflows:

    - ``stage_id``: the training stage that wrote this ledger.
    - ``dataset_revision``: the remote dataset prefix/version (e.g. ``"v2"``).
    - ``checkpoint_identity``: SHA-256 of the last committed checkpoint that
      was verified after writing shard completion metadata.
    - ``committed_optimizer_step``: the optimizer step number that was durably
      committed in the last checkpoint associated with this ledger state.
    - ``completed_metadata[shard_name]["token_chunks_trained"]``: how many
      token chunks were trained for this shard (for audit and bookkeeping).

    Completed prior-stage shards are skipped by default.  If identical content
    is intentionally reused in a later stage, the caller must pass
    ``allow_retrain=True`` to ``is_completed`` explicitly rather than silently
    retraining data.
    """
    # Ledger schema version written by this implementation.
    SCHEMA_VERSION = "3.0"

    def __init__(
        self,
        output_dir: str,
        stage_id: str = "",
        dataset_revision: str = "v2",
    ):
        self.output_dir = output_dir
        self.ledger_file = os.path.join(output_dir, "shard_ledger.json")
        self.schema_version: str = self.SCHEMA_VERSION
        self.stage_id: str = stage_id
        self.dataset_revision: str = dataset_revision
        self.completed_shards: List[str] = []
        self.completed_metadata: Dict[str, Any] = {}
        self.active_shard: Optional[str] = None
        self.active_shard_line_offset: int = 0
        self.active_shard_total_lines: int = 0
        self.active_shard_byte_offset: int = 0
        self.total_eligible_shards: int = 0
        self.active_shard_start_step: int = 0
        self.total_records_processed: int = 0
        self.last_committed_step: int = 0
        # Phase 3 fields — persisted for cross-session incremental integrity
        self.committed_optimizer_step: int = 0
        self.checkpoint_identity: str = ""
        self.domain_cursors: Dict[str, Dict[str, Any]] = {}
        self.active_domain_idx: int = 0
        self.load()

    def load(self):
        loaded_data = None
        if os.path.exists(self.ledger_file):
            try:
                with open(self.ledger_file, "r", encoding="utf-8") as f:
                    loaded_data = json.load(f)
                print(f"[Shard Ledger] Loaded ledger from local disk ({len(loaded_data.get('completed_shards', []))} completed shards).", flush=True)
            except Exception as e:
                print(f"[Shard Ledger Notice] Local ledger read note: {e}")

        if loaded_data:
            self.schema_version = loaded_data.get("schema_version", "1.0")
            # Accept stage_id and dataset_revision from disk only when the constructor
            # did not supply an explicit non-default value.
            if not self.stage_id:
                self.stage_id = loaded_data.get("stage_id", "")
            if self.dataset_revision == "v2":
                self.dataset_revision = loaded_data.get("dataset_revision", "v2")
            self.completed_shards = loaded_data.get("completed_shards", [])
            self.completed_metadata = loaded_data.get("completed_metadata", {})
            self.active_shard = loaded_data.get("active_shard")
            self.active_shard_line_offset = loaded_data.get("active_shard_line_offset", loaded_data.get("active_shard_offset", 0))
            self.active_shard_total_lines = loaded_data.get("active_shard_total_lines", 0)
            self.active_shard_byte_offset = loaded_data.get("active_shard_byte_offset", 0)
            self.total_eligible_shards = loaded_data.get("total_eligible_shards", 0)
            self.active_shard_start_step = loaded_data.get("active_shard_start_step", 0)
            self.total_records_processed = loaded_data.get("total_records_processed", 0)
            self.last_committed_step = loaded_data.get("last_committed_step", 0)
            # Phase 3 fields — restored for cross-session incremental integrity
            self.committed_optimizer_step = loaded_data.get("committed_optimizer_step", 0)
            self.checkpoint_identity = loaded_data.get("checkpoint_identity", "")
            self.domain_cursors = loaded_data.get("domain_cursors", {})
            self.active_domain_idx = loaded_data.get("active_domain_idx", 0)

    def save(self):
        os.makedirs(self.output_dir, exist_ok=True)
        data = {
            "schema_version": self.schema_version,
            "stage_id": self.stage_id,
            "dataset_revision": self.dataset_revision,
            "completed_shards": self.completed_shards,
            "completed_metadata": self.completed_metadata,
            "active_shard": self.active_shard,
            "active_shard_line_offset": self.active_shard_line_offset,
            "active_shard_total_lines": self.active_shard_total_lines,
            "active_shard_byte_offset": self.active_shard_byte_offset,
            "total_eligible_shards": self.total_eligible_shards,
            "active_shard_start_step": self.active_shard_start_step,
            "total_records_processed": self.total_records_processed,
            "last_committed_step": self.last_committed_step,
            "committed_optimizer_step": self.committed_optimizer_step,
            "checkpoint_identity": self.checkpoint_identity,
            "domain_cursors": self.domain_cursors,
            "active_domain_idx": self.active_domain_idx,
            "last_updated_utc": datetime.now(timezone.utc).isoformat(),
        }
        # Write to local temporary file first, then atomically move to authoritative local ledger file.
        # Checkpoints and ledgers write exclusively to local NVMe/SSD to guarantee zero-latency execution;
        tmp_dir = tempfile.gettempdir()
        tmp_ledger = os.path.join(tmp_dir, f"shard_ledger_{os.getpid()}_{int(time.time())}.json")
        try:
            with open(tmp_ledger, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            _stream_copy_file(tmp_ledger, self.ledger_file)
        finally:
            if os.path.exists(tmp_ledger):
                _safe_unlink(tmp_ledger)

    def get_total_completed_records(self) -> int:
        total = 0
        for meta in self.completed_metadata.values():
            total += int(meta.get("records_processed", 0))
        return total

    def is_completed(self, shard_name: str, local_path: Optional[str] = None, out_sha: Optional[dict] = None, verify_full_hash: bool = False) -> bool:
        """
        Fast authentic shard completion verification:
        Reuses verified recorded SHA-256 metadata when available to eliminate redundant multi-gigabyte re-hashing.
        """
        if shard_name not in self.completed_shards:
            return False
        recorded_sha = self.completed_metadata.get(shard_name, {}).get("file_sha256")
        if recorded_sha and out_sha is not None:
            out_sha["sha256"] = recorded_sha
        if local_path and os.path.exists(local_path):
            if os.path.getsize(local_path) == 0:
                return False
            if verify_full_hash and recorded_sha:
                cur_sha = compute_file_sha256(local_path)
                if out_sha is not None:
                    out_sha["sha256"] = cur_sha
                if cur_sha != recorded_sha:
                    print(
                        f"[Shard Ledger Verification Warning] Content hash mismatch for '{shard_name}'! "
                        f"(Ledger SHA: {recorded_sha[:16]}..., Current file SHA: {cur_sha[:16]}...). "
                        f"Shard content has changed; will not skip as completed!",
                        flush=True,
                    )
                    return False
        return True

    def reconcile_with_checkpoint(self, cursor: Dict[str, Any], step: int):
        """
        Reconciles ShardLedger state with authoritative checkpoint cursor.
        Ensures exact consistency if a crash occurred between checkpoint commit and ledger update.
        Handles both stale-behind and stale-ahead ledger anomalies.
        """
        c_shard = cursor.get("active_shard")
        c_offset = cursor.get("active_shard_line_offset", cursor.get("active_shard_offset", 0))
        c_byte_offset = cursor.get("active_shard_byte_offset", 0)
        c_records = cursor.get("total_samples_trained", cursor.get("total_completed_records", 0))
        c_completed = cursor.get("completed_shards")

        if c_completed is not None:
            checkpoint_completed_set = set(c_completed)

            # Evict any stale-ahead shards present in ledger but absent from checkpoint's completed list
            stale_ahead_shards = [s for s in self.completed_shards if s not in checkpoint_completed_set]
            if stale_ahead_shards:
                print(
                    f"[Shard Ledger Reconciliation] Evicting stale-ahead shards {stale_ahead_shards} from completed: "
                    f"verified checkpoint at step {step} has not committed these shards. They will be retrained.",
                    flush=True,
                )
                for s in stale_ahead_shards:
                    self.completed_metadata.pop(s, None)

            # Checkpoint's completed_shards is authoritative verified ground truth
            self.completed_shards = [s for s in c_completed if s != c_shard]
            self.completed_metadata = {
                k: v for k, v in self.completed_metadata.items()
                if k in checkpoint_completed_set and k != c_shard
            }
        else:
            if c_shard and c_shard in self.completed_shards:
                self.completed_shards = [s for s in self.completed_shards if s != c_shard]
                self.completed_metadata.pop(c_shard, None)

        if c_shard:
            self.active_shard = c_shard
            self.active_shard_line_offset = c_offset
            self.active_shard_byte_offset = c_byte_offset
            if c_shard in self.completed_shards:
                print(
                    f"[Shard Ledger Reconciliation] Unmarking active shard '{c_shard}' from completed: "
                    f"verified checkpoint only trained up to line {c_offset:,}.",
                    flush=True,
                )
                self.completed_shards = [s for s in self.completed_shards if s != c_shard]
                self.completed_metadata.pop(c_shard, None)

        if cursor.get("domain_cursors"):
            self.domain_cursors = copy.deepcopy(cursor.get("domain_cursors", {}))
        if "active_domain_idx" in cursor:
            self.active_domain_idx = cursor.get("active_domain_idx", 0)

        self.total_records_processed = c_records
        self.last_committed_step = step
        self.save()
        print(
            f"[Shard Ledger Reconciled] Authoritative state restored: "
            f"Active Shard='{self.active_shard}', Line Offset={self.active_shard_line_offset:,} (Byte Offset: {self.active_shard_byte_offset:,}), "
            f"Completed Shards={len(self.completed_shards)}, Domain Cursors={len(self.domain_cursors)}.",
            flush=True,
        )

    def update_domain_cursor(
        self,
        domain: str,
        active_shard: Optional[str],
        line_offset: int,
        byte_offset: int = 0,
        chunk_idx: int = 0,
        total_lines: int = 0,
    ):
        if not domain:
            return
        if domain not in self.domain_cursors:
            self.domain_cursors[domain] = {}
        self.domain_cursors[domain]["active_shard"] = active_shard
        self.domain_cursors[domain]["line_offset"] = line_offset
        self.domain_cursors[domain]["byte_offset"] = byte_offset
        self.domain_cursors[domain]["chunk_idx"] = chunk_idx
        if total_lines > 0:
            self.domain_cursors[domain]["total_lines"] = total_lines

    def update_active_cursor(
        self,
        shard_name: Optional[str],
        line_offset: int,
        total_samples: int,
        step: int,
        byte_offset: int = 0,
        shard_total_lines: int = 0,
        total_lines: Optional[int] = None,
    ):
        self.active_shard = shard_name
        self.active_shard_line_offset = line_offset
        self.active_shard_byte_offset = byte_offset
        effective_lines = shard_total_lines if shard_total_lines > 0 else (total_lines or 0)
        if effective_lines > 0:
            self.active_shard_total_lines = effective_lines
        self.total_records_processed = total_samples
        self.last_committed_step = step
        self.save()

        # Detailed Telemetry Log for Shard & Ledger Save Paths
        tot_lines = self.active_shard_total_lines
        pct_str = f" ({(line_offset / tot_lines) * 100:.2f}%)" if tot_lines > 0 else ""
        tot_lines_str = f"{tot_lines:,}" if tot_lines > 0 else "counting..."
        tot_shards_str = f"{len(self.completed_shards)}/{self.total_eligible_shards}" if self.total_eligible_shards > 0 else f"{len(self.completed_shards)}"

        print(
            f"[Shard Ledger Saved] Step {step:04d} | Active Shard: '{shard_name}' | "
            f"Line: {line_offset:,} / {tot_lines_str}{pct_str} | Byte Offset: {byte_offset:,} | "
            f"Completed Shards: {tot_shards_str}\n"
            f"  -> Local Ledger  : {os.path.abspath(self.ledger_file)}",
            flush=True,
        )

    def set_active_shard(self, shard_name: str, start_step: int = 0):
        if self.active_shard != shard_name:
            self.active_shard = shard_name
            self.active_shard_start_step = start_step
            self.save()

    def mark_completed(self, shard_name: str, records_count: int, file_sha256: str, token_chunks_trained: int = 0):
        if shard_name not in self.completed_shards:
            self.completed_shards.append(shard_name)
        self.completed_metadata[shard_name] = {
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "records_processed": records_count,
            "file_sha256": file_sha256,
            "token_chunks_trained": token_chunks_trained,
            "stage_id": self.stage_id,
        }
        self.total_records_processed = max(self.total_records_processed, self.get_total_completed_records())
        self.active_shard = None
        self.active_shard_line_offset = 0
        self.active_shard_byte_offset = 0
        self.save()
        print(f"[Shard Ledger COMPLETED] Shard '{shard_name}' registered as COMPLETED (SHA-256: {file_sha256[:16]}..., Records: {records_count:,}, Total: {len(self.completed_shards)} completed).", flush=True)

    def update_checkpoint_identity(self, checkpoint_sha: str, optimizer_step: int):
        """Record the last committed checkpoint identity and optimizer step for cross-session integrity."""
        self.checkpoint_identity = checkpoint_sha
        self.committed_optimizer_step = optimizer_step
        self.save()


# ---------------------------------------------------------------------------
# Persistent Shard Line Index (Task E)
# ---------------------------------------------------------------------------

class ShardLineIndex:
    """
    Content-hash-validated persistent line-byte-offset index for a JSONL shard.

    Stores ``{line_number: byte_offset}`` pairs so that intra-shard resumption
    can seek directly to the target line without a sequential scan.  The index
    is invalidated and rebuilt whenever the shard's SHA-256 changes, ensuring
    that a modified shard (same filename, different content) is never resumed
    at a stale byte position.

    Index files are stored under ``<dataset_dir>/.tara_idx/<shard_sha256>.idx.json``
    and are treated as expendable operational cache — they are rebuilt on demand
    and never modify the shard files themselves.
    """

    def __init__(self, dataset_dir: str):
        self.idx_dir = os.path.join(dataset_dir, ".tara_idx")

    def _index_path(self, shard_sha256: str) -> str:
        return os.path.join(self.idx_dir, f"{shard_sha256}.idx.json")

    def load(self, shard_sha256: str) -> Optional[Dict[int, int]]:
        """
        Load the byte-offset index for the given shard SHA-256.

        Returns a ``{line_idx: byte_offset}`` dict if a valid index exists,
        or ``None`` if the index is absent or unreadable.  The index key is the
        SHA-256 of the shard content, so an index built for a different content
        revision is automatically rejected.
        """
        idx_path = self._index_path(shard_sha256)
        if not os.path.exists(idx_path):
            return None
        try:
            with open(idx_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            # JSON keys are always strings; convert back to int line numbers.
            return {int(k): int(v) for k, v in raw.items()}
        except Exception as e:
            print(f"[ShardLineIndex] Index read error for SHA {shard_sha256[:16]}...: {e}", flush=True)
            return None

    def build_and_save(self, shard_path: str, shard_sha256: str) -> Dict[int, int]:
        """
        Build a byte-offset index by scanning ``shard_path`` once, then persist
        it so subsequent resumes skip the scan. Only applicable to text JSONL files.
        Binary formats (.parquet, .arrow, etc.) are skipped.
        """
        ext = os.path.splitext(shard_path)[1].lower()
        if ext in (".parquet", ".arrow", ".bin"):
            return {}
        offsets: Dict[int, int] = {}
        try:
            with open(shard_path, "rb") as f:
                line_idx = 0
                while True:
                    pos = f.tell()
                    line = f.readline()
                    if not line:
                        break
                    offsets[line_idx] = pos
                    line_idx += 1
            os.makedirs(self.idx_dir, exist_ok=True)
            idx_path = self._index_path(shard_sha256)
            tmp_path = idx_path + f".tmp_{os.getpid()}"
            with open(tmp_path, "w", encoding="utf-8") as f:
                # Store keys as strings (JSON requirement); load() converts back.
                json.dump({str(k): v for k, v in offsets.items()}, f)
            try:
                os.replace(tmp_path, idx_path)
            except Exception:
                shutil.copy2(tmp_path, idx_path)
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
            print(f"[ShardLineIndex] Built and saved index for SHA {shard_sha256[:16]}... ({len(offsets):,} lines).", flush=True)
        except Exception as e:
            print(f"[ShardLineIndex] Index build error for '{shard_path}': {e}", flush=True)
        return offsets

    def get_or_build(self, shard_path: str, shard_sha256: str) -> Dict[int, int]:
        """
        Return a valid byte-offset index, loading from disk if available or
        building on demand if the cached index is absent or stale.
        """
        ext = os.path.splitext(shard_path)[1].lower()
        if ext in (".parquet", ".arrow", ".bin"):
            return {}
        cached = self.load(shard_sha256)
        if cached is not None:
            return cached
        return self.build_and_save(shard_path, shard_sha256)


# ---------------------------------------------------------------------------
# Dataset Snapshot — deterministic shard catalog with live content tracking
# ---------------------------------------------------------------------------

class DatasetSnapshot:
    """
    Deterministic, persistent snapshot of the dataset shard catalog.

    Records the authoritative ordered list of shard identities for a training
    run — including repo identity, prefix/version, ordered shard filenames, and
    runtime-computed content SHA-256 values as each shard is acquired.

    **What this is NOT:**
    The Hugging Face dataset repository does not publish a separate signed manifest
    file.  This class does NOT invent a remote manifest format.  Instead it uses
    the API response from ``HfApi.list_repo_files()`` (or a local glob) as the
    discovery source, and computes content hashes at the moment each shard is
    downloaded or opened locally — storing them as the single authoritative record
    of what content was actually trained on.

    **Shard change detection:**
    - Added shards: present in the live catalog but absent from the snapshot.
    - Removed shards: present in the snapshot but absent from the live catalog.
    - Renamed shards: treated as add + remove (no cross-name SHA tracking).
    - Content-modified shards: same name, live SHA differs from snapshot SHA.
      A content-modified shard is NOT treated as completed — it will be retrained.

    **Persistence:** Stored at ``<snapshot_dir>/dataset_snapshot.json``.
    Written atomically (tmp → rename) to survive crashes.
    """

    SCHEMA_VERSION = "1.0"

    def __init__(self, snapshot_dir: str):
        self.snapshot_path = os.path.join(snapshot_dir, "dataset_snapshot.json")
        self.schema_version: str = self.SCHEMA_VERSION
        self.repo_id: str = ""
        self.prefix: str = ""
        self.revision: Optional[str] = None
        self.discovered_at_utc: Optional[str] = None
        # Ordered list of shard filenames as originally discovered
        self.ordered_shards: List[str] = []
        # Runtime-computed content SHA-256 per shard (populated as shards are opened)
        self.shard_content_hashes: Dict[str, str] = {}
        self.load()

    def load(self):
        if not os.path.exists(self.snapshot_path):
            return
        try:
            with open(self.snapshot_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.schema_version = data.get("schema_version", "1.0")
            self.repo_id = data.get("repo_id", "")
            self.prefix = data.get("prefix", "")
            self.revision = data.get("revision")
            self.discovered_at_utc = data.get("discovered_at_utc")
            self.ordered_shards = data.get("ordered_shards", [])
            self.shard_content_hashes = data.get("shard_content_hashes", {})
        except Exception as e:
            print(f"[DatasetSnapshot] Load error (non-fatal): {e}", flush=True)

    def save(self):
        os.makedirs(os.path.dirname(self.snapshot_path), exist_ok=True)
        data = {
            "schema_version": self.schema_version,
            "repo_id": self.repo_id,
            "prefix": self.prefix,
            "revision": self.revision,
            "discovered_at_utc": self.discovered_at_utc,
            "ordered_shards": self.ordered_shards,
            "shard_content_hashes": self.shard_content_hashes,
            "last_updated_utc": datetime.now(timezone.utc).isoformat(),
        }
        tmp = self.snapshot_path + f".tmp_{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        try:
            os.replace(tmp, self.snapshot_path)
        except Exception:
            shutil.copy2(tmp, self.snapshot_path)
            try:
                os.remove(tmp)
            except Exception:
                pass

    def initialize_from_discovery(self, repo_id: str, prefix: str, ordered_shards: List[str], revision: Optional[str] = None):
        """
        Record the initial shard catalog as returned by the discovery step.
        Only updates the catalog when the ordered shard list has changed.
        A failed or incomplete/empty discovery cannot overwrite a valid prior snapshot.
        """
        if not ordered_shards and self.ordered_shards:
            print("[DatasetSnapshot Warning] Refusing to overwrite valid prior shard snapshot with empty discovery result.", flush=True)
            return

        self.repo_id = repo_id
        self.prefix = prefix
        self.revision = revision
        if self.ordered_shards != ordered_shards:
            self.ordered_shards = list(ordered_shards)
            self.discovered_at_utc = datetime.now(timezone.utc).isoformat()
            self.save()

    def record_shard_hash(self, shard_name: str, sha256: str):
        """
        Record the runtime-computed content SHA-256 for a shard.
        Called immediately after ``compute_file_sha256`` in ``__iter__``.
        """
        if self.shard_content_hashes.get(shard_name) != sha256:
            self.shard_content_hashes[shard_name] = sha256
            self.save()

    def diff_against_live(self, live_shards: List[str]) -> Dict[str, List[str]]:
        """
        Compare the persisted ordered catalog against a fresh live discovery.

        Returns:
          {
            "added":   shards in live but not in snapshot,
            "removed": shards in snapshot but not in live,
            "common":  shards present in both (may have content changes),
          }
        """
        snap_set = set(self.ordered_shards)
        live_set = set(live_shards)
        return {
            "added":   sorted(live_set - snap_set),
            "removed": sorted(snap_set - live_set),
            "common":  sorted(snap_set & live_set),
        }

    def is_shard_content_changed(self, shard_name: str, current_sha: str) -> bool:
        """
        Returns True when a shard's live content hash differs from the recorded hash.
        Returns False when no prior hash exists (first time seen).
        """
        recorded = self.shard_content_hashes.get(shard_name)
        if recorded is None:
            return False  # Not yet recorded — not a change, just new
        return recorded != current_sha


class TaraStreamingJsonlDataset(IterableDataset):
    """
    High-Throughput Zero-Memory Streaming Dataset.

    Streams JSONL records on-the-fly directly to the GPU without loading
    millions of samples into RAM. Supports Just-In-Time single shard downloading,
    persistent ledger tracking, exact deterministic line cursor resumption,
    and controlled shard completion/eviction that is driven by the training loop
    (not by this iterator) so that shard completion is never recorded ahead of a
    durable, verified checkpoint.

    Token chunking policy (Task B):
    - Every document whose token sequence exceeds max_seq_len is split into
      consecutive, non-overlapping chunks of exactly max_seq_len tokens.
    - Each chunk is identified by a stable (shard_name, line_idx, chunk_idx) triple
      that is preserved in the yielded item and persisted in the checkpoint cursor,
      enabling exact resumption.
    - The final short chunk of a document (len < max_seq_len) is yielded as-is when
      it contains at least 4 tokens; otherwise it is discarded.  This policy is
      explicit and tested.
    - Loss/label masking from the original document (target_start) is applied
      independently per chunk: tokens that belong to the input portion of the chunk
      are masked with -100 so they do not contribute to the language-modelling loss.
    - No token from any intended training document is silently truncated or dropped.
    """
    def __init__(
        self,
        file_paths: List[str],
        tokenizer: Any,
        max_seq_len: int = 256,
        max_samples: int = 0,
        shuffle_shards: bool = False,
        resume_shard: Optional[str] = None,
        resume_line_offset: int = -1,
        resume_byte_offset: int = 0,
        resume_chunk_idx: int = 0,
        shard_ledger: Optional[ShardLedger] = None,
        keep_shards: bool = False,
        dataset_dir: Optional[str] = None,
        repo_id: Optional[str] = None,
        prefix: str = "v2",
        buffer_size: int = 1,
        max_shard_bytes: int = 0,
        dataset_snapshot: Optional["DatasetSnapshot"] = None,
    ):
        self.file_paths = list(file_paths)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_samples = max_samples
        self.shuffle_shards = shuffle_shards
        self.resume_shard = resume_shard
        self.resume_line_offset = resume_line_offset
        self.resume_byte_offset = resume_byte_offset
        self.resume_chunk_idx = resume_chunk_idx
        self.shard_ledger = shard_ledger
        self.keep_shards = keep_shards
        self.dataset_dir = dataset_dir
        self.repo_id = repo_id
        self.prefix = prefix
        self.max_shard_bytes = max(0, int(max_shard_bytes))
        self.managed_downloaded_shards: Set[str] = set()
        # Persistent byte-offset index for fast intra-shard resume (Task E / Phase 3.1-A).
        # Only instantiated when a local dataset directory is available.
        self._line_index: Optional[ShardLineIndex] = ShardLineIndex(dataset_dir) if dataset_dir else None
        # Dataset snapshot for content-change detection across sessions.
        self._snapshot: Optional[DatasetSnapshot] = dataset_snapshot
        # Strict deterministic line cursor streaming: buffer_size = 1 guarantees zero duplicate and zero lost samples.
        # buffer_size > 1 introduces per-shard random shuffling of items in a rolling window.  When a resume cursor
        # is active (resume_shard is set), items from multiple positions are mixed in the buffer before being yielded,
        # which makes it impossible to guarantee that the cursor position recorded in the checkpoint corresponds to
        # the last sample actually consumed by the optimizer.  This is a known data-ordering limitation: resumed
        # training with buffer_size > 1 may replay or omit a window of samples equal in size to buffer_size around
        # the resume boundary.  For exact deterministic resumption, buffer_size must be 1.
        self.buffer_size = max(1, int(buffer_size))
        if self.buffer_size > 1 and resume_shard is not None:
            print(
                f"[Dataset Resume Warning] buffer_size={self.buffer_size} with an active resume cursor "
                f"(resume_shard='{resume_shard}') produces non-deterministic cursor positions inside the "
                f"shuffle window.  Exact-once resumption is not guaranteed for up to {self.buffer_size} "
                f"samples around the resume boundary.  Set buffer_size=1 for fully deterministic resumption.",
                flush=True,
            )
        self.current_active_shard: Optional[str] = None
        self.active_shard_name: Optional[str] = None
        self.active_shard_offset: int = 0
        self.active_shard_total_lines: int = 0
        self.active_shard_byte_offset: int = 0
        self.active_chunk_idx: int = 0

        # Shards that have been fully streamed but whose completion must be
        # confirmed by the training loop after a durable checkpoint is committed.
        # Maps shard_name -> (total_shard_lines, shard_sha256).
        self._pending_shard_completions: Dict[str, Tuple[int, str]] = {}

        # Validate resume target existence if explicitly specified
        if self.resume_shard:
            norm_resume = self.resume_shard.replace("\\", "/")
            base_resume = os.path.basename(norm_resume)
            known_shard_names = set()
            for p in self.file_paths:
                norm_p = p.replace("\\", "/")
                known_shard_names.add(norm_p)
                known_shard_names.add(os.path.basename(norm_p))
                if self.dataset_dir:
                    norm_ds = self.dataset_dir.replace("\\", "/").rstrip("/")
                    if norm_p.startswith(norm_ds + "/"):
                        known_shard_names.add(norm_p[len(norm_ds) + 1:])
            if norm_resume not in known_shard_names and base_resume not in known_shard_names:
                raise RuntimeError(
                    f"[Dataset Resume Error] Specified resume target shard '{self.resume_shard}' "
                    f"is not present in the discovered dataset shards catalog ({len(known_shard_names)} shards)."
                )

        print(f"[Dataset] Initialized streaming pipeline across {len(self.file_paths)} shard(s) "
              f"(buffer_size={self.buffer_size}, resume_shard={self.resume_shard}, "
              f"line_offset={self.resume_line_offset}, byte_offset={self.resume_byte_offset}, "
              f"chunk_idx={self.resume_chunk_idx})...")

    def commit_shard_completion(self, shard_name: str, managed_cache: bool = False, actual_fp: Optional[str] = None):
        """
        Marks a shard as completed in the ledger and optionally evicts its managed
        cache copy.  Must be called by the training loop ONLY after a durable,
        verified checkpoint has been committed that covers all training units from
        the shard.  Calling this from inside __iter__ is strictly forbidden.
        """
        if shard_name not in self._pending_shard_completions:
            return
        total_lines, shard_sha = self._pending_shard_completions.pop(shard_name)
        if self.shard_ledger:
            self.shard_ledger.mark_completed(shard_name, total_lines, shard_sha)
        if not self.keep_shards and managed_cache and actual_fp and os.path.exists(actual_fp):
            try:
                os.remove(actual_fp)
                self.managed_downloaded_shards.discard(os.path.abspath(actual_fp))
                print(f"[JIT Shard Eviction] Reclaimed disk space: deleted managed completed shard cache {shard_name}", flush=True)
            except Exception as evict_err:
                print(f"[JIT Shard Eviction Notice] Could not delete managed cache {shard_name}: {evict_err}")

    @staticmethod
    def _extract_training_text(record: dict) -> Tuple[str, str, str]:
        """
        Extracts (inp_text, out_text, raw_text) from heterogeneous records across all TARA domains:
        - OpenWebMath, FineWeb, Caselaw: text / content
        - COCO Captions: sentences_raw / caption
        - FLEURS Audio: transcription / raw_transcription / sentence
        - BFCL Tool-Use: question / function
        - Lean 4 Formal Math: formal_statement / proof / goal
        - CommitPackFT: commit / diff / message
        - Standard SFT: input / output / prompt / response / messages
        """
        # 1. Text or Content
        if "text" in record and record["text"]:
            return "", "", str(record["text"])
        if "content" in record and record["content"]:
            return "", "", str(record["content"])

        # 2. Input / Output or Prompt / Response
        if "input" in record or "output" in record:
            return str(record.get("input", "")), str(record.get("output", "")), ""
        if "prompt" in record or "response" in record:
            return str(record.get("prompt", "")), str(record.get("response", "")), ""

        # 3. Vision Captions (COCO has 'sentences_raw' as list)
        if "sentences_raw" in record and record["sentences_raw"]:
            sents = record["sentences_raw"]
            cap = sents[0] if isinstance(sents, list) and sents else str(sents)
            return "", "", f"<|vision_start|> [Vision Context] <|vision_end|> {str(cap)}"
        if "caption" in record and record["caption"]:
            cap = record["caption"]
            cap_txt = cap[0] if isinstance(cap, list) and cap else str(cap)
            return "", "", f"<|vision_start|> [Vision Context] <|vision_end|> {str(cap_txt)}"

        # 4. Audio Transcriptions (FLEURS)
        if "transcription" in record and record["transcription"]:
            return "", "", f"<|audio_start|> [Audio Transcription] <|audio_end|> {str(record['transcription'])}"
        if "sentence" in record and record["sentence"]:
            return "", "", f"<|audio_start|> [Audio Transcription] <|audio_end|> {str(record['sentence'])}"

        # 5. Tool Use (BFCL has 'question' and 'function')
        if "question" in record and record["question"]:
            q = record["question"]
            funcs = json.dumps(record.get("function", []))
            user_msg = ""
            if isinstance(q, list):
                for turn in q:
                    if isinstance(turn, list):
                        for m in turn:
                            if isinstance(m, dict) and "content" in m:
                                user_msg += str(m["content"]) + " "
                    elif isinstance(turn, dict) and "content" in turn:
                        user_msg += str(turn["content"]) + " "
                    else:
                        user_msg += str(turn) + " "
            else:
                user_msg = str(q)
            return f"<|tara_exec|> Tools: {funcs}\nQuery: {user_msg.strip()}", str(record.get("id", "")), ""

        # 6. Formal Math Proofs (Lean 4)
        if "formal_statement" in record or "proof" in record:
            stmt = str(record.get("formal_statement", record.get("goal", "")))
            prf = str(record.get("proof", record.get("code", "")))
            return f"Formal Statement:\n{stmt}", f"Proof:\n{prf}", ""

        # 7. Code Diffs & Bug Fixes (CommitPackFT)
        if "commit" in record or "diff" in record or "new_file" in record:
            msg = str(record.get("message", record.get("subject", "")))
            diff = str(record.get("diff", record.get("new_file", "")))[:2048]
            return f"Fix commit: {msg}", diff, ""

        # 8. Chat Messages format
        if "messages" in record and isinstance(record["messages"], list):
            conv = []
            for m in record["messages"]:
                if isinstance(m, dict):
                    r = m.get("role", "user")
                    c = m.get("content", "")
                    conv.append(f"<|im_start|>{r}\n{c}<|im_end|>")
            if conv:
                return "", "", "\n".join(conv)

        # 9. Fallback: stringify non-empty scalar values
        vals = [str(v) for v in record.values() if isinstance(v, (str, int, float)) and str(v).strip()]
        if vals:
            return "", "", " ".join(vals[:5])
        return "", "", ""

    def _iter_shard_records(
        self,
        actual_fp: str,
        is_resume_target: bool,
        target_resume_line: int,
        target_resume_byte: int,
        target_resume_chunk: int,
        shard_name: str,
        shard_sha: str,
    ):
        ext = os.path.splitext(actual_fp)[1].lower()

        def should_skip_record(current_idx: int) -> bool:
            if is_resume_target:
                if current_idx < target_resume_line:
                    return True
                if current_idx == target_resume_line and target_resume_chunk == 0:
                    return True
            return False

        # Check for Git-LFS pointer stub
        try:
            with open(actual_fp, "rb") as check_f:
                header = check_f.read(100)
                if b"version https://git-lfs.github.com/spec/v1" in header:
                    print(f"[Dataset Shard Error] '{actual_fp}' is an unresolved Git-LFS pointer. Skipping raw stub.", flush=True)
                    return
        except Exception:
            pass

        # Branch 1: Parquet files (.parquet)
        if ext == ".parquet":
            try:
                import pyarrow.parquet as pq
                pf = pq.ParquetFile(actual_fp)
                line_idx = 0
                for batch in pf.iter_batches(batch_size=64):
                    df = batch.to_pydict()
                    keys = list(df.keys())
                    n_rows = len(df[keys[0]]) if keys else 0
                    for r in range(n_rows):
                        current_idx = line_idx
                        line_idx += 1
                        if should_skip_record(current_idx):
                            continue
                        row = {k: df[k][r] for k in keys}
                        yield current_idx, (current_idx + 1) * 1024, row
                return
            except Exception as pe:
                print(f"[Parquet Stream Error] {actual_fp}: {pe}", flush=True)
                return

        # Branch 2: Compressed Gzip (.gz / .jsonl.gz / .txt.gz)
        if ext in (".gz", ".tar.gz"):
            import gzip
            try:
                with gzip.open(actual_fp, "rt", encoding="utf-8", errors="ignore") as gf:
                    line_idx = 0
                    for gline in gf:
                        current_idx = line_idx
                        line_idx += 1
                        if should_skip_record(current_idx):
                            continue
                        gline = gline.strip()
                        if not gline:
                            continue
                        try:
                            rec = json.loads(gline)
                        except Exception:
                            rec = {"text": gline}
                        yield current_idx, (current_idx + 1) * 1024, rec
                return
            except Exception as ge:
                print(f"[Gzip Stream Error] {actual_fp}: {ge}", flush=True)
                return

        # Branch 3: Delimited CSV / TSV
        if ext in (".csv", ".tsv"):
            import csv
            delim = "\t" if ext == ".tsv" else ","
            try:
                with open(actual_fp, "r", encoding="utf-8", errors="ignore") as cf:
                    reader = csv.reader(cf, delimiter=delim)
                    line_idx = 0
                    header_row = None
                    for row_cells in reader:
                        if not row_cells:
                            continue
                        # Detect FLEURS TSV with no headers (col[1] is .wav, col[2] is transcription)
                        if len(row_cells) >= 3 and row_cells[1].endswith((".wav", ".mp3", ".flac")):
                            current_idx = line_idx
                            line_idx += 1
                            if should_skip_record(current_idx):
                                continue
                            yield current_idx, (current_idx + 1) * 1024, {"transcription": row_cells[2]}
                            continue

                        if header_row is None:
                            header_row = [c.strip() for c in row_cells]
                            continue

                        current_idx = line_idx
                        line_idx += 1
                        if should_skip_record(current_idx):
                            continue
                        row_dict = {header_row[i]: row_cells[i] for i in range(min(len(header_row), len(row_cells)))}
                        yield current_idx, (current_idx + 1) * 1024, row_dict
                return
            except Exception as ce:
                print(f"[CSV/TSV Stream Error] {actual_fp}: {ce}", flush=True)
                return

        # Branch 4: Plain Text (.txt / .text)
        if ext in (".txt", ".text"):
            try:
                with open(actual_fp, "r", encoding="utf-8", errors="ignore") as tf:
                    buf = []
                    line_idx = 0
                    for tline in tf:
                        tline = tline.strip()
                        if not tline:
                            if buf:
                                current_idx = line_idx
                                line_idx += 1
                                if not (is_resume_target and current_idx < target_resume_line):
                                    yield current_idx, (current_idx + 1) * 1024, {"text": "\n".join(buf)}
                                buf = []
                        else:
                            buf.append(tline)
                            if len(buf) >= 40:
                                current_idx = line_idx
                                line_idx += 1
                                if not (is_resume_target and current_idx < target_resume_line):
                                    yield current_idx, (current_idx + 1) * 1024, {"text": "\n".join(buf)}
                                buf = []
                    if buf:
                        current_idx = line_idx
                        if not (is_resume_target and current_idx < target_resume_line):
                            yield current_idx, (current_idx + 1) * 1024, {"text": "\n".join(buf)}
                return
            except Exception as te:
                print(f"[Text Stream Error] {actual_fp}: {te}", flush=True)
                return

        # Branch 5: JSON file (check if array or lines)
        if ext == ".json":
            try:
                with open(actual_fp, "r", encoding="utf-8", errors="ignore") as jf:
                    content_start = jf.read(50).strip()
                    jf.seek(0)
                    if content_start.startswith("["):
                        data_arr = json.load(jf)
                        for line_idx, rec in enumerate(data_arr):
                            if is_resume_target and line_idx < target_resume_line:
                                continue
                            yield line_idx, (line_idx + 1) * 1024, rec
                        return
            except Exception as je:
                pass

        # Branch 6: Standard JSONL (with ShardLineIndex fast byte-seek resume support)
        with open(actual_fp, "r", encoding="utf-8", errors="ignore") as f:
            start_line_idx = 0
            if is_resume_target and (target_resume_line >= 0 or target_resume_chunk > 0):
                sought = False
                file_end = f.seek(0, 2)
                f.seek(0)
                if self._line_index is not None:
                    try:
                        offsets = self._line_index.get_or_build(actual_fp, shard_sha)
                        if target_resume_chunk == 0:
                            next_line = target_resume_line + 1
                            if next_line in offsets and 0 < offsets[next_line] <= file_end:
                                f.seek(offsets[next_line])
                                start_line_idx = next_line
                                sought = True
                            elif 0 < target_resume_byte <= file_end:
                                f.seek(target_resume_byte)
                                start_line_idx = next_line
                                sought = True
                        else:
                            if target_resume_line in offsets and 0 <= offsets[target_resume_line] <= file_end:
                                f.seek(offsets[target_resume_line])
                                start_line_idx = target_resume_line
                                sought = True
                            elif target_resume_line == 0:
                                f.seek(0)
                                start_line_idx = 0
                                sought = True
                    except Exception as idx_err:
                        print(f"[ShardLineIndex] Seek error: {idx_err}. Falling back to sequential.", flush=True)
                        f.seek(0)
                        start_line_idx = 0

                if not sought:
                    if target_resume_chunk == 0 and 0 < target_resume_byte <= file_end:
                        f.seek(target_resume_byte)
                        start_line_idx = target_resume_line + 1
                        sought = True

                if sought:
                    print(
                        f"[Dataset Stream] Byte-seek resume: shard '{shard_name}' "
                        f"positioned at byte {f.tell():,} (line {start_line_idx:,}, "
                        f"chunk_idx_resume={target_resume_chunk}).",
                        flush=True,
                    )
                else:
                    f.seek(0)
                    start_line_idx = 0

            line_offset_in_read = 0
            while True:
                line = f.readline()
                if not line:
                    break
                line_idx = start_line_idx + line_offset_in_read
                line_offset_in_read += 1
                line_byte_offset = f.tell()
                line = line.strip()
                if not line:
                    continue

                if is_resume_target and start_line_idx == 0:
                    if line_idx < target_resume_line:
                        continue
                    if line_idx == target_resume_line and target_resume_chunk == 0:
                        continue

                try:
                    record = json.loads(line)
                    yield line_idx, line_byte_offset, record
                except Exception:
                    continue

    def __iter__(self):
        fps = list(self.file_paths)
        if self.shuffle_shards:
            r = random.Random(42)
            r.shuffle(fps)

        emitted = 0
        buffer = []
        buffer_size = self.buffer_size
        target_resume_shard = self.resume_shard.replace("\\", "/") if self.resume_shard else None
        target_resume_line = self.resume_line_offset
        target_resume_byte = getattr(self, "resume_byte_offset", 0)
        target_resume_chunk = getattr(self, "resume_chunk_idx", 0)

        # Track whether resume target has been reached in sequential ordering
        resume_target_reached = (target_resume_shard is None)

        if target_resume_shard:
            print(f"[Dataset Stream] Resuming exact data cursor: shard '{target_resume_shard}', "
                  f"line offset > {target_resume_line:,} (byte offset: {target_resume_byte:,}, "
                  f"chunk_idx >= {target_resume_chunk})...", flush=True)

        for fp in fps:
            # Preserve normalized relative path to avoid basename collisions across subfolders
            shard_name = fp.replace("\\", "/")
            if self.dataset_dir:
                norm_ds = self.dataset_dir.replace("\\", "/").rstrip("/")
                if shard_name.startswith(norm_ds + "/"):
                    shard_name = shard_name[len(norm_ds) + 1:]

            # Skip earlier shards BEFORE invoking the downloader when resuming from a middle shard
            if not resume_target_reached:
                if shard_name == target_resume_shard or (target_resume_shard and os.path.basename(shard_name) == os.path.basename(target_resume_shard)):
                    resume_target_reached = True
                else:
                    # Earlier shard precedes resume target; skip without downloading!
                    continue

            is_resume_target = (shard_name == target_resume_shard or bool(target_resume_shard and os.path.basename(shard_name) == os.path.basename(target_resume_shard)))

            # Determine authentic local file path
            if os.path.isabs(fp) or os.path.exists(fp):
                actual_fp = fp
            elif self.dataset_dir:
                actual_fp = os.path.join(self.dataset_dir, shard_name)
            else:
                actual_fp = fp

            # Check Shard Ledger: Skip if already marked completed.
            # Pass out_sha so the ledger's verification SHA is captured and reused below,
            # eliminating a redundant second read of the same file.
            _ledger_sha_cache: dict = {}
            _shard_file_for_ledger = actual_fp if os.path.exists(actual_fp) else None
            if self.shard_ledger and self.shard_ledger.is_completed(shard_name, local_path=_shard_file_for_ledger, out_sha=_ledger_sha_cache):
                if not is_resume_target:
                    print(f"[Dataset Ledger] Shard '{shard_name}' is ALREADY COMPLETED and verified. Skipping.", flush=True)
                    continue
                else:
                    # Active resume target is already marked completed in the ledger.
                    # Verify if the cursor indicates that all lines were processed.
                    meta = self.shard_ledger.completed_metadata.get(shard_name, {})
                    rec_count = meta.get("records_processed", 0)
                    if rec_count > 0 and target_resume_line >= rec_count - 1:
                        print(
                            f"[Dataset Ledger] Resume target shard '{shard_name}' is verified COMPLETED in ledger "
                            f"({rec_count:,} records, cursor line: {target_resume_line:,}). "
                            f"Skipping unnecessary download and advancing to subsequent shard.",
                            flush=True,
                        )
                        target_resume_shard = None
                        target_resume_line = -1
                        target_resume_byte = 0
                        target_resume_chunk = 0
                        resume_target_reached = True
                        continue

            # If local file missing or 0 bytes, download on-demand if remote repo configured
            if not os.path.exists(actual_fp) or os.path.getsize(actual_fp) == 0:
                if self.repo_id and self.dataset_dir:
                    actual_fp = download_single_shard_jit(self.repo_id, self.prefix, shard_name, self.dataset_dir,
                                                          max_shard_bytes=self.max_shard_bytes)
                    self.managed_downloaded_shards.add(os.path.abspath(actual_fp))
                else:
                    continue

            # Reuse SHA computed during ledger verification above; only compute fresh if no SHA was
            # captured (ledger miss, no recorded SHA, or file was just downloaded after the ledger check).
            shard_sha = _ledger_sha_cache.get("sha256") or compute_file_sha256(actual_fp)
            self.active_shard_name = shard_name
            self.current_active_shard = shard_name

            # Count records without creating a huge byte-offset dictionary.
            self.active_shard_total_lines = 0
            ext = os.path.splitext(actual_fp)[1].lower()
            if ext == ".parquet":
                try:
                    import pyarrow.parquet as pq
                    self.active_shard_total_lines = int(
                        pq.ParquetFile(actual_fp).metadata.num_rows
                    )
                except Exception as count_err:
                    print(
                        f"[Parquet Row Count Warning] {actual_fp}: {count_err}",
                        flush=True,
                    )
            elif ext in (".jsonl", ".ndjson") and self._line_index is not None and self._line_index.load(shard_sha) is not None:
                self.active_shard_total_lines = len(self._line_index.load(shard_sha))
            else:
                # Format-aware ultra-fast buffered binary line counter for .jsonl, .txt, .csv, and compressed .gz
                try:
                    if ext in (".gz", ".tar.gz") or actual_fp.endswith(".gz"):
                        import gzip
                        with gzip.open(actual_fp, "rb") as gf:
                            self.active_shard_total_lines = sum(
                                chunk.count(b"\n") for chunk in iter(lambda: gf.read(1024 * 1024), b"")
                            )
                    elif ext in (".jsonl", ".ndjson", ".txt", ".text", ".csv", ".tsv"):
                        with open(actual_fp, "rb") as f:
                            self.active_shard_total_lines = sum(
                                chunk.count(b"\n") for chunk in iter(lambda: f.read(1024 * 1024), b"")
                            )
                except Exception:
                    pass
            if self.shard_ledger:
                self.shard_ledger.active_shard_total_lines = (
                    self.active_shard_total_lines
                )
                self.shard_ledger.set_active_shard(shard_name)

            # Record live content hash in the dataset snapshot so content changes are detected
            # across sessions without re-discovering the full shard list.
            if self._snapshot is not None:
                self._snapshot.record_shard_hash(shard_name, shard_sha)

            print(
                f"[Dataset Shard Active] Streaming from: {shard_name} "
                f"(Total Lines: {self.active_shard_total_lines:,}, Runtime SHA-256: {shard_sha[:16]}...)",
                flush=True,
            )

            total_shard_lines = 0
            reached_eof = True
            try:
                for line_idx, line_byte_offset, record in self._iter_shard_records(
                    actual_fp, is_resume_target, target_resume_line, target_resume_byte, target_resume_chunk, shard_name, shard_sha
                ):
                    total_shard_lines = max(total_shard_lines, line_idx + 1)
                    try:
                        inp_text, out_text, raw_text = self._extract_training_text(record)
                        if not inp_text and not out_text and not raw_text:
                            continue

                        if not inp_text and not out_text and raw_text:
                            enc = self.tokenizer.encode(raw_text)
                            full_ids = enc.ids if hasattr(enc, "ids") else list(enc)
                            target_start = 0
                        elif inp_text or out_text:
                            enc_i = self.tokenizer.encode(inp_text)
                            enc_o = self.tokenizer.encode(out_text)
                            enc_inp = enc_i.ids if hasattr(enc_i, "ids") else list(enc_i)
                            enc_out = enc_o.ids if hasattr(enc_o, "ids") else list(enc_o)
                            target_start = len(enc_inp)
                            full_ids = enc_inp + enc_out
                        else:
                            continue

                        # Task B: Deterministic consecutive chunking.
                        n_tokens = len(full_ids)
                        if n_tokens < 4:
                            continue

                        chunk_size = self.max_seq_len
                        n_chunks = max(1, math.ceil(n_tokens / chunk_size))

                        valid_chunks = []
                        for chunk_idx in range(n_chunks):
                            chunk_start = chunk_idx * chunk_size
                            chunk_end = min(chunk_start + chunk_size, n_tokens)
                            chunk_ids = full_ids[chunk_start:chunk_end]
                            if len(chunk_ids) >= 4:
                                valid_chunks.append((chunk_idx, chunk_ids, chunk_start))

                        for c_i, (chunk_idx, chunk_ids, chunk_start) in enumerate(valid_chunks):
                            is_last_chunk = (c_i == len(valid_chunks) - 1)

                            if is_resume_target and line_idx == target_resume_line and chunk_idx < target_resume_chunk:
                                continue

                            chunk_target_start = max(0, target_start - chunk_start)
                            chunk_target_start = min(chunk_target_start, len(chunk_ids))

                            input_ids = torch.tensor(chunk_ids, dtype=torch.long)
                            labels = input_ids.clone()
                            if chunk_target_start > 0:
                                labels[:chunk_target_start] = -100

                            self.active_shard_offset = line_idx
                            self.active_shard_byte_offset = line_byte_offset
                            self.active_chunk_idx = chunk_idx

                            item = (input_ids, labels, shard_name, line_idx, line_byte_offset, chunk_idx, is_last_chunk)
                            if buffer_size <= 1:
                                yield item
                                emitted += 1
                                if 0 < self.max_samples <= emitted:
                                    reached_eof = False
                                    return
                            else:
                                buffer.append(item)
                                if len(buffer) >= buffer_size:
                                    idx = random.randrange(len(buffer))
                                    yield buffer.pop(idx)
                                    emitted += 1
                                    if 0 < self.max_samples <= emitted:
                                        reached_eof = False
                                        return
                    except Exception:
                        continue

                # Drain remaining buffer for this shard before signalling EOF
                while buffer:
                    idx = random.randrange(len(buffer)) if buffer_size > 1 else 0
                    yield buffer.pop(idx)
                    emitted += 1
                    if 0 < self.max_samples <= emitted:
                        reached_eof = False
                        return

                # -------------------------------------------------------
                # Task C: Shard EOF reached — do NOT mark completed or
                # evict here.  Instead, register this shard in
                # _pending_shard_completions so that the training loop
                # can call commit_shard_completion() after a durable
                # verified checkpoint has been committed.
                # -------------------------------------------------------
                if reached_eof:
                    is_managed_cache = (os.path.abspath(actual_fp) in self.managed_downloaded_shards)
                    self._pending_shard_completions[shard_name] = (total_shard_lines, shard_sha)
                    print(
                        f"[Dataset Shard EOF] Shard '{shard_name}' streaming complete "
                        f"({total_shard_lines:,} lines). Completion pending checkpoint commit.",
                        flush=True,
                    )

                if 0 < self.max_samples <= emitted:
                    break
            except Exception as read_err:
                print(f"[Dataset Shard Error] Error reading {shard_name}: {read_err}", flush=True)
                continue
            finally:
                if is_resume_target:
                    target_resume_shard = None
                    target_resume_line = -1
                    target_resume_byte = 0
                    target_resume_chunk = 0
                    self.resume_shard = None
                    self.resume_line_offset = -1
                    self.resume_byte_offset = 0
                    self.resume_chunk_idx = 0

        while buffer:
            idx = random.randrange(len(buffer)) if buffer_size > 1 else 0
            yield buffer.pop(idx)
            emitted += 1
            if 0 < self.max_samples <= emitted:
                return


# Backward-compatibility alias
TaraJsonlDataset = TaraStreamingJsonlDataset


def extract_state_transition(rec: Any) -> Optional[Tuple[str, str, str]]:
    """
    Extracts genuine environment state transitions (s_t, a_t) -> s_{t+1} from:
    1. Direct keys: state_observation / state / observation, action / intervention, next_state_observation / next_state
    2. Embedded derivation text: e.g. State observation: ... Action: ... Next state observation: ...
    3. Curriculum causal transitions (e.g. agi_causal_transitions records)
    4. Parquet / QA records (e.g. 08_commonsense_world_model)
    """
    if not isinstance(rec, dict):
        return None

    t_obj = rec.get("state_transition") if isinstance(rec.get("state_transition"), dict) else rec
    s_obs = t_obj.get("state_observation") or t_obj.get("state") or t_obj.get("observation")
    act = t_obj.get("action") or t_obj.get("intervention")
    ns_obs = t_obj.get("next_state_observation") or t_obj.get("next_state") or t_obj.get("subsequent_state")

    if (
        isinstance(s_obs, str) and s_obs.strip()
        and isinstance(act, str) and act.strip()
        and isinstance(ns_obs, str) and ns_obs.strip()
    ):
        reward = t_obj.get("reward")
        if reward is not None and not isinstance(reward, (int, float)):
            return None
        confidence = t_obj.get("confidence")
        if confidence is not None:
            if not isinstance(confidence, (int, float)) or confidence < 0.0 or confidence > 1.0:
                return None
        return (s_obs.strip(), act.strip(), ns_obs.strip())

    if any(k in t_obj for k in ("state_observation", "next_state_observation")):
        return None

    deriv = str(rec.get("derivation", ""))
    if "State observation" in deriv or "State s" in deriv:
        m = re.search(
            r"State(?:\s+observation)?:\s*(.+?)\.\s*Action:\s*(.+?)\.\s*Next(?:\s+state)?(?:\s+observation)?:\s*(.+?)(?:\.\s*Reward:|$)",
            deriv,
            re.DOTALL | re.IGNORECASE,
        )
        if m:
            s_val, a_val, ns_val = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
            if s_val and a_val and ns_val:
                return (s_val, a_val, ns_val)

    if (
        rec.get("domain") == "agi_causal_transitions"
        or "transition" in str(rec.get("concept", "")).lower()
        or "transition" in str(rec.get("topic", "")).lower()
        or "transition" in str(rec.get("id", "")).lower()
    ):
        s = str(rec.get("definition", rec.get("concept", rec.get("input", ""))))[:512].strip()
        a = str(rec.get("formula", rec.get("subject", "causal_transition")))[:256].strip()
        ns = str(rec.get("derivation", rec.get("explanation", rec.get("output", ""))))[:1024].strip()
        if s and a and ns:
            return (s, a, ns)

    if rec.get("question") and (rec.get("answer") or rec.get("response")):
        s = str(rec.get("question"))[:512].strip()
        a = str(rec.get("action", "reasoning_transition"))[:256].strip()
        ns = str(rec.get("answer", rec.get("response"))).strip()
        if s and ns:
            return (s, a, ns)

    if rec.get("input") and rec.get("output"):
        s = str(rec.get("input"))[:512].strip()
        a = str(rec.get("action", "transition"))[:256].strip()
        ns = str(rec.get("output"))[:1024].strip()
        if s and ns:
            return (s, a, ns)

    return None


def inspect_candidate_world_transitions(fp: str, max_records: int = 50) -> int:
    """Inspects a candidate world model transition file (JSONL or Parquet) and returns count of valid transitions."""
    if not os.path.isfile(fp):
        return 0
    count = 0
    ext = os.path.splitext(fp)[1].lower()
    if ext == ".parquet":
        try:
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(fp)
            for batch in pf.iter_batches(batch_size=min(max_records, 64)):
                df = batch.to_pydict()
                keys = list(df.keys())
                n_rows = len(df[keys[0]]) if keys else 0
                for r in range(n_rows):
                    row = {k: df[k][r] for k in keys}
                    if extract_state_transition(row) is not None:
                        count += 1
                        if count >= max_records:
                            return count
                if count >= max_records:
                    break
        except Exception:
            return count
    else:
        try:
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                for line_idx, line in enumerate(f):
                    if line_idx >= max_records * 2:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        if extract_state_transition(rec) is not None:
                            count += 1
                            if count >= max_records:
                                return count
                    except Exception:
                        continue
        except Exception:
            return count
    return count


def route_dataset_record(rec: Any) -> str:
    """
    Categorizes a dataset record into its designated processing pathway:
      - 'transition': genuine environment dynamics with (s_t, a_t, s_{t+1})
      - 'knowledge_fact': discrete knowledge graph triples / relational entities
      - 'neural_text': natural language text, instructions, or conversations
      - 'invalid': malformed, unparseable, or non-compliant records
    Strictly prevents schema conflation: knowledge triples are never treated as transitions.
    """
    if not isinstance(rec, dict):
        return "invalid"

    # Support optional nested "state_transition" wrapper
    t_obj = rec.get("state_transition") if isinstance(rec.get("state_transition"), dict) else rec

    # 1. Genuine environment state transition via unified extractor
    if extract_state_transition(rec) is not None:
        return "transition"

    if any(k in t_obj for k in ("state_observation", "next_state_observation")):
        return "invalid"

    # 2. Knowledge graph / relational facts pathway
    knowledge_keys = ("subject", "predicate", "relation", "triples", "entities", "ontology_node", "world_fact")
    if any(k in rec for k in knowledge_keys):
        subj = rec.get("subject")
        pred = rec.get("predicate") or rec.get("relation")
        obj = rec.get("object")
        if (
            (isinstance(subj, str) and subj.strip() and isinstance(pred, str) and pred.strip() and isinstance(obj, str) and obj.strip())
            or rec.get("triples") or rec.get("entities") or rec.get("world_fact")
        ):
            return "knowledge_fact"

    # 3. Neural sequence modeling pathway
    neural_keys = ("prompt", "instruction", "input", "output", "completion", "text", "conversation", "messages")
    if any(k in rec for k in neural_keys):
        text_val = rec.get("text") or rec.get("instruction") or rec.get("prompt") or rec.get("input")
        if text_val is not None:
            return "neural_text"

    return "invalid"


class TaraDomainBatchItem(tuple):
    """
    Multi-domain batch item that acts simultaneously as:
    1. A standard 7-tuple: (input_ids, labels, shard_name, line_idx, byte_offset, chunk_idx, is_last_chunk)
       compatible with existing dataset collate_fn and 7-tuple unpacking.
    2. A dictionary with keys 'input_ids', 'labels', 'attention_mask', 'domain', 'shard_name', etc.
    3. Property access for .input_ids, .labels, .domain, etc.
    """
    def __new__(cls, input_ids, labels, shard_name, line_idx, byte_offset, chunk_idx, is_last, domain="general"):
        instance = super().__new__(cls, (input_ids, labels, shard_name, line_idx, byte_offset, chunk_idx, is_last))
        instance._domain = domain
        return instance

    @property
    def input_ids(self): return self[0]
    @property
    def labels(self): return self[1]
    @property
    def attention_mask(self): return (self[0] != 0).long() if isinstance(self[0], torch.Tensor) else None
    @property
    def shard_name(self): return self[2]
    @property
    def line_idx(self): return self[3]
    @property
    def byte_offset(self): return self[4]
    @property
    def chunk_idx(self): return self[5]
    @property
    def is_last(self): return self[6]
    @property
    def domain(self): return getattr(self, "_domain", "general")

    def __contains__(self, key):
        if isinstance(key, str):
            return key in {
                "input_ids", "labels", "attention_mask", "shard_name",
                "line_idx", "byte_offset", "chunk_idx", "is_last",
                "domain"
            }
        return super().__contains__(key)

    def __getitem__(self, key):
        if isinstance(key, str):
            mapping = {
                "input_ids": self[0],
                "labels": self[1],
                "attention_mask": (self[0] != 0).long() if isinstance(self[0], torch.Tensor) else None,
                "shard_name": self[2],
                "line_idx": self[3],
                "byte_offset": self[4],
                "chunk_idx": self[5],
                "is_last": self[6],
                "domain": getattr(self, "_domain", "general"),
            }
            if key in mapping:
                return mapping[key]
            raise KeyError(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


class WorldModelTransitionDataset(IterableDataset):
    """
    Streaming dataset for genuine World Model state transitions: (s_t, a_t) -> s_{t+1}.
    Supports both JSONL and Parquet formatted transitions, curriculum causal transitions,
    and prompt-target state action modeling.
    Knowledge-graph facts and triples are routed to the knowledge graph pathway,
    and are strictly rejected as transition supervision.
    """
    def __init__(
        self,
        file_paths: List[str],
        tokenizer: Any,
        max_seq_len: int = 256,
        max_samples: int = 0,
    ):
        self.file_paths = list(file_paths)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_samples = max_samples

    def _encode_transition(self, s_obs: str, act: str, ns_obs: str) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Encodes state transition into prompt-target sequence with prompt masked in labels."""
        prompt_text = f"[STATE] {s_obs} [ACTION] {act} [NEXT_STATE] "
        target_text = f"{ns_obs} [EOS]"
        enc_p = self.tokenizer.encode(prompt_text)
        prompt_tokens = enc_p.ids if hasattr(enc_p, "ids") else (enc_p if isinstance(enc_p, (list, tuple)) else list(enc_p))
        enc_t = self.tokenizer.encode(target_text)
        target_tokens = enc_t.ids if hasattr(enc_t, "ids") else (enc_t if isinstance(enc_t, (list, tuple)) else list(enc_t))
        if not target_tokens:
            return None
        if len(prompt_tokens) >= self.max_seq_len - 4:
            prompt_tokens = prompt_tokens[-(self.max_seq_len // 2):]
        full_ids = (prompt_tokens + target_tokens)[:self.max_seq_len]
        if len(full_ids) < 4:
            return None
        input_ids = torch.tensor(full_ids, dtype=torch.long)
        labels = input_ids.clone()
        mask_len = min(len(prompt_tokens), len(full_ids))
        labels[:mask_len] = -100
        if not (labels != -100).any():
            return None
        return input_ids, labels

    def __iter__(self):
        emitted = 0
        for fp in self.file_paths:
            if not os.path.exists(fp):
                continue
            ext = os.path.splitext(fp)[1].lower()
            if ext == ".parquet":
                try:
                    import pyarrow.parquet as pq
                    pf = pq.ParquetFile(fp)
                    for batch in pf.iter_batches(batch_size=64):
                        df = batch.to_pydict()
                        keys = list(df.keys())
                        n_rows = len(df[keys[0]]) if keys else 0
                        for r in range(n_rows):
                            row = {k: df[k][r] for k in keys}
                            if route_dataset_record(row) != "transition":
                                continue
                            t = extract_state_transition(row)
                            if not t:
                                continue
                            encoded = self._encode_transition(*t)
                            if encoded is None:
                                continue
                            input_ids, labels = encoded
                            yield TaraDomainBatchItem(input_ids, labels, os.path.basename(fp), emitted, 0, 0, True, domain="08_commonsense_world_model")
                            emitted += 1
                            if 0 < self.max_samples <= emitted:
                                return
                except Exception as e:
                    print(f"[World Dataset Parquet Error] {fp}: {e}", flush=True)
            else:
                try:
                    with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                        for line_idx, line in enumerate(f):
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                rec = json.loads(line)
                            except Exception:
                                continue
                            if not isinstance(rec, dict):
                                continue
                            if route_dataset_record(rec) != "transition":
                                continue
                            t = extract_state_transition(rec)
                            if not t:
                                continue
                            encoded = self._encode_transition(*t)
                            if encoded is None:
                                continue
                            input_ids, labels = encoded
                            yield TaraDomainBatchItem(input_ids, labels, os.path.basename(fp), line_idx, 0, 0, True, domain="08_commonsense_world_model")
                            emitted += 1
                            if 0 < self.max_samples <= emitted:
                                return
                except Exception as e:
                    print(f"[World Dataset Error] Failed to stream {fp}: {e}", flush=True)


# =============================================================================
# 3. Multi-Domain Interleaved Streaming, World Model & Replay Buffers
# =============================================================================

CANONICAL_DOMAINS = [
    "01_mathematics_formal_logic",
    "02_computer_science_systems",
    "03_natural_sciences_stem",
    "04_linguistics_multilingual",
    "05_philosophy_cognitive_ethics",
    "06_social_sciences_economics_law",
    "07_applied_engineering_robotics",
    "08_commonsense_world_model",
    "09_multimodal_audio",
    "10_multimodal_vision",
    "11_tool_use_execution",
    "12_planning_reasoning_traces",
    "13_debugging_self_correction",
    "instruction_tuning_sft",
]

DOMAIN_ALIASES = {
    "01_math": "01_mathematics_formal_logic",
    "math": "01_mathematics_formal_logic",
    "mathematics": "01_mathematics_formal_logic",
    "01_mathematics_formal_logic": "01_mathematics_formal_logic",
    "02_code": "02_computer_science_systems",
    "code": "02_computer_science_systems",
    "cs": "02_computer_science_systems",
    "computer_science": "02_computer_science_systems",
    "02_computer_science_systems": "02_computer_science_systems",
    "03_stem": "03_natural_sciences_stem",
    "stem": "03_natural_sciences_stem",
    "natural_sciences": "03_natural_sciences_stem",
    "03_natural_sciences_stem": "03_natural_sciences_stem",
    "04_kannada": "04_linguistics_multilingual",
    "kannada": "04_linguistics_multilingual",
    "linguistics": "04_linguistics_multilingual",
    "indic": "04_linguistics_multilingual",
    "multilingual": "04_linguistics_multilingual",
    "04_linguistics_multilingual": "04_linguistics_multilingual",
    "05_philosophy": "05_philosophy_cognitive_ethics",
    "philosophy": "05_philosophy_cognitive_ethics",
    "ethics": "05_philosophy_cognitive_ethics",
    "05_philosophy_cognitive_ethics": "05_philosophy_cognitive_ethics",
    "06_social": "06_social_sciences_economics_law",
    "social": "06_social_sciences_economics_law",
    "economics": "06_social_sciences_economics_law",
    "law": "06_social_sciences_economics_law",
    "06_social_sciences_economics_law": "06_social_sciences_economics_law",
    "07_engineering": "07_applied_engineering_robotics",
    "engineering": "07_applied_engineering_robotics",
    "robotics": "07_applied_engineering_robotics",
    "07_applied_engineering_robotics": "07_applied_engineering_robotics",
    "08_world_model": "08_commonsense_world_model",
    "world_model": "08_commonsense_world_model",
    "commonsense": "08_commonsense_world_model",
    "08_commonsense_world_model": "08_commonsense_world_model",
    "09_audio": "09_multimodal_audio",
    "audio": "09_multimodal_audio",
    "multimodal_audio": "09_multimodal_audio",
    "09_multimodal_audio": "09_multimodal_audio",
    "10_vision": "10_multimodal_vision",
    "vision": "10_multimodal_vision",
    "multimodal_vision": "10_multimodal_vision",
    "10_multimodal_vision": "10_multimodal_vision",
    "11_tool_use": "11_tool_use_execution",
    "tools": "11_tool_use_execution",
    "tool": "11_tool_use_execution",
    "tool_use": "11_tool_use_execution",
    "11_tool_use_execution": "11_tool_use_execution",
    "12_reasoning": "12_planning_reasoning_traces",
    "reasoning": "12_planning_reasoning_traces",
    "planning": "12_planning_reasoning_traces",
    "traces": "12_planning_reasoning_traces",
    "12_planning_reasoning_traces": "12_planning_reasoning_traces",
    "13_debugging": "13_debugging_self_correction",
    "debugging": "13_debugging_self_correction",
    "self_correction": "13_debugging_self_correction",
    "13_debugging_self_correction": "13_debugging_self_correction",
    "sft": "instruction_tuning_sft",
    "instruction": "instruction_tuning_sft",
    "instruction_tuning": "instruction_tuning_sft",
    "instruction_tuning_sft": "instruction_tuning_sft",
}


def resolve_shard_domain(shard_path: str) -> str:
    """
    Resolves a shard path to its authentic canonical domain across arbitrary nested subfolders
    (e.g., domain/folder/folder/folder/shard.parquet or v2/domain/sub/shard.jsonl).
    Uses hierarchical directory segment matching and longest-alias matching to prevent
    short alias collisions (such as 'cs' in 'linguistics' or 'physics').
    """
    norm = shard_path.replace("\\", "/").strip("/")
    parts = [p.lower() for p in norm.split("/") if p]
    dir_parts = parts[:-1]  # directory segments excluding filename

    # 1. Exact directory segment matching against canonical domains & aliases
    for p in dir_parts:
        if p in DOMAIN_ALIASES:
            return DOMAIN_ALIASES[p]
        if p in CANONICAL_DOMAINS:
            return p

    # 2. Prefix matching on directory segments (e.g. '01_math' or '04_kannada')
    for p in dir_parts:
        for d in CANONICAL_DOMAINS:
            if p.startswith(d) or d.startswith(p):
                return d
        for alias, canon in DOMAIN_ALIASES.items():
            if p.startswith(alias) or p.endswith(alias):
                return canon

    # 3. Whole path search with word boundaries or segment search
    for d in CANONICAL_DOMAINS:
        if f"/{d}/" in f"/{norm}/" or norm.startswith(d):
            return d

    # 4. Longest alias match in path to prevent short aliases like 'cs' stealing from 'linguistics'
    sorted_aliases = sorted(DOMAIN_ALIASES.items(), key=lambda x: len(x[0]), reverse=True)
    for alias, canon in sorted_aliases:
        if len(alias) >= 4 and alias in norm.lower():
            return canon

    # 5. Dynamic fallback: if directory exists and is not a generic wrapper, use it as a dynamic domain
    for p in dir_parts:
        if p not in ("v1", "v2", "v3", "data", "dataset", "datasets", "storage", "downloaded", "main_dataset_v2"):
            return p

    return "02_computer_science_systems"


def parse_domain_weights(
    weight_str: str,
    active_domains: Optional[List[str]] = None,
    available_domains: Optional[List[str]] = None,
    **kwargs,
) -> Dict[str, float]:
    """
    Parses flexible comma-separated domain weight mixing (e.g.
    '01_math:0.20,02_code:0.30' or '01_math=0.20,02_code=0.30').
    Normalizes weights over target domains so they sum strictly to 1.0.
    """
    domains = list(active_domains) if active_domains is not None else (list(available_domains) if available_domains is not None else list(CANONICAL_DOMAINS))
    if not domains:
        domains = list(CANONICAL_DOMAINS)

    weights = {}
    if weight_str:
        for item in weight_str.split(","):
            item = item.strip()
            if not item:
                continue
            sep = ":" if ":" in item else ("=" if "=" in item else None)
            if not sep:
                continue
            k, v = item.split(sep, 1)
            k = k.strip().lower()
            try:
                val = float(v.strip())
            except ValueError:
                continue
            canon = None
            if k in domains:
                canon = k
            elif DOMAIN_ALIASES.get(k) in domains:
                canon = DOMAIN_ALIASES.get(k)
            else:
                canon = DOMAIN_ALIASES.get(k)
                if not canon:
                    for d in domains:
                        if d.startswith(k) or k in d:
                            canon = d
                            break
                if not canon:
                    for d in CANONICAL_DOMAINS:
                        if d.startswith(k) or k in d:
                            canon = d
                            break
            if canon:
                weights[canon] = max(0.0, val)

    if not weights:
        default_weights = {
            "02_computer_science_systems": 0.25,
            "01_mathematics_formal_logic": 0.15,
            "04_linguistics_multilingual": 0.15,
            "12_planning_reasoning_traces": 0.15,
            "03_natural_sciences_stem": 0.10,
            "08_commonsense_world_model": 0.05,
            "11_tool_use_execution": 0.05,
            "13_debugging_self_correction": 0.02,
            "instruction_tuning_sft": 0.02,
            "05_philosophy_cognitive_ethics": 0.012,
            "06_social_sciences_economics_law": 0.012,
            "07_applied_engineering_robotics": 0.012,
            "09_multimodal_audio": 0.012,
            "10_multimodal_vision": 0.012,
        }
        for d in domains:
            weights[d] = default_weights.get(d, 0.01)

    assigned_sum = sum(weights.get(d, 0.0) for d in domains)
    unassigned = [d for d in domains if d not in weights]
    if unassigned:
        remaining = max(0.0, 1.0 - assigned_sum)
        per_domain = remaining / len(unassigned) if remaining > 0 else 0.01
        for d in unassigned:
            weights[d] = per_domain

    total = sum(weights.get(d, 0.0) for d in domains)
    if total > 0:
        return {d: weights.get(d, 0.0) / total for d in domains}
    return {d: 1.0 / len(domains) for d in domains}


class TaraMultiDomainScheduler(IterableDataset):
    """
    Weighted Multi-Domain Interleaved Streaming Scheduler with Deficit Round-Robin (DRR).
    
    Eliminates catastrophic forgetting by interleaving batches from 14 domain buckets
    proportional to domain weights, ensuring core reasoning, math, and linguistics
    remain active throughout large-scale pretraining.
    """
    def __init__(
        self,
        domain_shards: Dict[str, List[str]],
        domain_weights: Optional[Dict[str, float]] = None,
        tokenizer: Optional[Any] = None,
        batch_size: int = 1,
        max_seq_len: int = 256,
        max_samples: int = 0,
        dataset_dir: Optional[str] = None,
        repo_id: Optional[str] = None,
        prefix: str = "",
        shard_ledger: Optional[ShardLedger] = None,
        keep_shards: bool = False,
        buffer_size: int = 1,
        max_shard_bytes: int = 0,
        dataset_snapshot: Optional[Any] = None,
        domain_cursors: Optional[Dict[str, Dict[str, Any]]] = None,
        **kwargs,
    ):
        self.domain_shards = {d: list(s) for d, s in domain_shards.items() if s}
        self.active_domains = sorted(self.domain_shards.keys())
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.max_samples = max_samples
        self.dataset_dir = dataset_dir
        self.repo_id = repo_id
        self.prefix = prefix
        self.shard_ledger = shard_ledger
        self.keep_shards = keep_shards
        self.buffer_size = buffer_size
        self.max_shard_bytes = max_shard_bytes
        self.dataset_snapshot = dataset_snapshot

        self.domain_weights = parse_domain_weights("", self.active_domains)
        if domain_weights:
            norm_w = {}
            for d in self.active_domains:
                if d in domain_weights:
                    norm_w[d] = float(domain_weights[d])
            total_w = sum(norm_w.values())
            if total_w > 0:
                self.domain_weights = {d: norm_w.get(d, 0.0) / total_w for d in self.active_domains}

        self.deficits: Dict[str, float] = {d: 0.0 for d in self.active_domains}
        self.domain_cursors: Dict[str, Dict[str, Any]] = copy.deepcopy(domain_cursors) if domain_cursors else {}
        self.active_domain_idx: int = 0

        self.current_active_shard: Optional[str] = None
        self.active_shard_offset: int = 0
        self.active_shard_byte_offset: int = 0
        self.active_shard_total_lines: int = 0
        self.active_chunk_idx: int = 0
        self.active_domain: str = ""

        self.domain_datasets: Dict[str, TaraStreamingJsonlDataset] = {}
        for d in self.active_domains:
            cur = self.domain_cursors.get(d, {})
            self.domain_datasets[d] = TaraStreamingJsonlDataset(
                file_paths=self.domain_shards[d],
                tokenizer=tokenizer,
                max_seq_len=max_seq_len,
                max_samples=max_samples,
                shuffle_shards=False,
                resume_shard=cur.get("active_shard"),
                resume_line_offset=cur.get("line_offset", -1),
                resume_byte_offset=cur.get("byte_offset", 0),
                resume_chunk_idx=cur.get("chunk_idx", 0),
                shard_ledger=shard_ledger,
                keep_shards=keep_shards,
                dataset_dir=dataset_dir,
                repo_id=repo_id,
                prefix=prefix,
                buffer_size=buffer_size,
                max_shard_bytes=max_shard_bytes,
                dataset_snapshot=dataset_snapshot,
            )

    @property
    def _pending_shard_completions(self) -> Dict[str, Tuple[int, str]]:
        merged = {}
        for ds in self.domain_datasets.values():
            merged.update(getattr(ds, "_pending_shard_completions", {}))
        return merged

    @property
    def managed_downloaded_shards(self) -> Set[str]:
        merged = set()
        for ds in self.domain_datasets.values():
            merged.update(getattr(ds, "managed_downloaded_shards", set()))
        return merged

    def commit_shard_completion(self, shard_name: str, managed_cache: bool = False, actual_fp: Optional[str] = None):
        for ds in self.domain_datasets.values():
            if shard_name in getattr(ds, "_pending_shard_completions", {}):
                ds.commit_shard_completion(shard_name, managed_cache=managed_cache, actual_fp=actual_fp)
                return
        if self.shard_ledger:
            self.shard_ledger.mark_completed(shard_name, 0, "")

    def get_state(self) -> Dict[str, Any]:
        return {
            "domain_cursors": copy.deepcopy(self.domain_cursors),
            "active_domain_idx": self.active_domain_idx,
            "scheduler_deficits": copy.deepcopy(self.deficits),
        }

    def load_state(self, state: Dict[str, Any]):
        if not state:
            return
        if "domain_cursors" in state:
            self.domain_cursors = copy.deepcopy(state["domain_cursors"])
            for d, cur in self.domain_cursors.items():
                if d in self.domain_shards:
                    self.domain_datasets[d] = TaraStreamingJsonlDataset(
                        file_paths=self.domain_shards[d],
                        tokenizer=self.tokenizer,
                        max_seq_len=self.max_seq_len,
                        max_samples=self.max_samples,
                        shuffle_shards=False,
                        resume_shard=cur.get("active_shard"),
                        resume_line_offset=cur.get("line_offset", -1),
                        resume_byte_offset=cur.get("byte_offset", 0),
                        resume_chunk_idx=cur.get("chunk_idx", 0),
                        shard_ledger=self.shard_ledger,
                        keep_shards=self.keep_shards,
                        dataset_dir=self.dataset_dir,
                        repo_id=self.repo_id,
                        prefix=self.prefix,
                        buffer_size=self.buffer_size,
                        max_shard_bytes=self.max_shard_bytes,
                        dataset_snapshot=self.dataset_snapshot,
                    )
        if "active_domain_idx" in state:
            self.active_domain_idx = state["active_domain_idx"]
        if "scheduler_deficits" in state:
            self.deficits = copy.deepcopy(state["scheduler_deficits"])

    def __iter__(self):
        emitted = 0
        domain_iters = {d: iter(self.domain_datasets[d]) for d in self.active_domains}
        active_candidates = list(self.active_domains)

        while active_candidates:
            for d in active_candidates:
                self.deficits[d] += self.domain_weights.get(d, 0.0)

            best_d = max(active_candidates, key=lambda d: self.deficits[d])
            self.deficits[best_d] -= 1.0

            try:
                raw_item = next(domain_iters[best_d])
                input_ids, labels, shard_name, line_idx, byte_offset, chunk_idx, is_last = raw_item

                item = TaraDomainBatchItem(
                    input_ids, labels, shard_name, line_idx, byte_offset, chunk_idx, is_last,
                    domain=best_d
                )

                self.active_domain = best_d
                self.current_active_shard = shard_name
                self.active_shard_offset = line_idx
                self.active_shard_byte_offset = byte_offset
                self.active_chunk_idx = chunk_idx
                self.active_shard_total_lines = getattr(self.domain_datasets[best_d], "active_shard_total_lines", 0)

                if best_d not in self.domain_cursors:
                    self.domain_cursors[best_d] = {}
                self.domain_cursors[best_d]["active_shard"] = shard_name
                self.domain_cursors[best_d]["line_offset"] = line_idx
                self.domain_cursors[best_d]["byte_offset"] = byte_offset
                self.domain_cursors[best_d]["chunk_idx"] = chunk_idx
                self.domain_cursors[best_d]["total_lines"] = self.active_shard_total_lines

                if self.shard_ledger:
                    self.shard_ledger.update_domain_cursor(
                        domain=best_d,
                        active_shard=shard_name,
                        line_offset=line_idx,
                        byte_offset=byte_offset,
                        chunk_idx=chunk_idx,
                        total_lines=self.active_shard_total_lines,
                    )

                yield item
                emitted += 1
                if 0 < self.max_samples <= emitted:
                    return

            except StopIteration:
                if best_d in active_candidates:
                    active_candidates.remove(best_d)


def collate_fn(batch):
    if not batch:
        return torch.empty((0, 0), dtype=torch.long), torch.empty((0, 0), dtype=torch.long), []

    max_len = max(len(b[0]) for b in batch)
    bsz = len(batch)
    padded_inputs = torch.zeros(bsz, max_len, dtype=torch.long)
    padded_labels = torch.full((bsz, max_len), -100, dtype=torch.long)
    cursors = []

    for i, item in enumerate(batch):
        inp, lab = item[0], item[1]
        l = len(inp)
        padded_inputs[i, :l] = inp
        padded_labels[i, :l] = lab
        # item layout: (input_ids, labels, shard_name, line_idx, byte_offset, chunk_idx, is_last_chunk, [domain])
        dom = getattr(item, "domain", getattr(item, "_domain", "general"))
        if len(item) >= 8:
            cursors.append((item[2], item[3], item[4], item[5], item[6], item[7]))
        elif len(item) >= 7:
            cursors.append((item[2], item[3], item[4], item[5], item[6], dom))
        elif len(item) >= 6:
            cursors.append((item[2], item[3], item[4], item[5], False, dom))
        elif len(item) >= 5:
            cursors.append((item[2], item[3], item[4], 0, False, dom))
        elif len(item) >= 4:
            cursors.append((item[2], item[3], 0, 0, False, dom))
        else:
            cursors.append(("", 0, 0, 0, False, dom))

    return padded_inputs, padded_labels, cursors


# =============================================================================
# 4. Storage & Local Checkpoint Engine
# =============================================================================

def load_candidate_weights(model: nn.Module, model_dir: str) -> bool:
    """Loads existing candidate safetensors weights if present shard-by-shard to minimize CPU memory footprint."""
    safetensors_files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not safetensors_files or not HAS_SAFETENSORS:
        return False

    print(f"[Weights] Ingesting {len(safetensors_files)} existing SafeTensors shard(s) from {model_dir}...")
    model_sd = model.state_dict()
    matched = 0
    for sf in safetensors_files:
        shard_tensors = load_file(sf, device="cpu")
        for k, v in shard_tensors.items():
            if k in model_sd and model_sd[k].shape == v.shape:
                model_sd[k].copy_(v.to(dtype=model_sd[k].dtype))
                matched += 1
        del shard_tensors
        gc.collect()

    del model_sd
    gc.collect()
    print(f"[Weights] Successfully restored {matched} parameter tensors.")
    return matched > 0





