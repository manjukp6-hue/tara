#!/usr/bin/env python3
"""
Dedicated Standalone Neural World Model GPU Training Pipeline.

Trains the Latent Dynamics Transition Engine, Neural Reward Head,
Failure Probability Head, and Value Function on GPU.

Outputs:
  - 'world_model_dynamics.safetensors'
  - 'world_model_config.json'
  - 'parity_evaluation.json' (for cross-language Rust verification)
"""

import argparse
import json
import math
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from safetensors.torch import save_file as save_safetensors
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class DynamicsTransformerLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        rms_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.gqa_ratio = max(1, num_attention_heads // num_key_value_heads)

        q_dim = num_attention_heads * head_dim
        kv_dim = num_key_value_heads * head_dim

        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.q_proj = nn.Linear(hidden_size, q_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, kv_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, kv_dim, bias=False)
        self.o_proj = nn.Linear(q_dim, hidden_size, bias=False)

        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # Pre-LN Self-Attention with GQA over [State, Action] tokens (T=2)
        norm_h1 = self.input_layernorm(h)
        b, seq_len, _ = h.shape

        q = self.q_proj(norm_h1).view(b, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(norm_h1).view(b, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(norm_h1).view(b, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if self.gqa_ratio > 1:
            k = k.repeat_interleave(self.gqa_ratio, dim=1)
            v = v.repeat_interleave(self.gqa_ratio, dim=1)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale
        attn_weights = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn_weights, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(b, seq_len, -1)

        h = h + self.o_proj(attn_out)

        # Pre-LN SwiGLU / GELU Feed-Forward Network for both tokens
        norm_h2 = self.post_attention_layernorm(h)
        gate = self.gate_proj(norm_h2)
        up = self.up_proj(norm_h2)
        ffn_act = F.gelu(gate, approximate="tanh") * up
        h = h + self.down_proj(ffn_act)

        return h


class NeuralWorldModel(nn.Module):
    """
    Native Transformer Latent Dynamics World Model.
    100% Parameter parity with native Rust 'WorldModel' in tara_engine:
      - State/Action projection + RMSNorm
      - N Transformer Dynamics Blocks (Pre-LN, GQA Self-Attention, SwiGLU/GELU FFN)
      - Final RMSNorm
      - Continuous Residual Latent Transition: s_{t+1} = s_t + tanh(W_next h)
      - Reward, Failure Viability, and Expected Value heads
    """
    def __init__(
        self,
        latent_dim: int = 2048,
        action_dim: int = 512,
        hidden_size: int = 2048,
        intermediate_size: int = 5632,
        num_hidden_layers: int = 8,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        head_dim: int = 64,
        rms_norm_eps: float = 1e-5,
        discount_factor: float = 0.99,
        learning_rate: float = 0.0003,
        horizon_limit: int = 64,
    ):
        super().__init__()
        assert head_dim * num_attention_heads == hidden_size, "head_dim * num_heads must equal hidden_size"

        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.rms_norm_eps = rms_norm_eps
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.horizon_limit = horizon_limit

        # 1. Input Projections & Input Norm
        self.state_proj = nn.Linear(latent_dim, hidden_size, bias=False)
        self.action_proj = nn.Linear(action_dim, hidden_size, bias=False)
        self.input_norm = RMSNorm(hidden_size, eps=rms_norm_eps)

        # 2. Transformer Latent Dynamics Blocks
        self.layers = nn.ModuleList([
            DynamicsTransformerLayer(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
                rms_norm_eps=rms_norm_eps,
            )
            for _ in range(num_hidden_layers)
        ])

        # 3. Final Backbone Norm
        self.norm = RMSNorm(hidden_size, eps=rms_norm_eps)

        # 4. Continuous Residual Latent Transition Head
        self.next_state_proj = nn.Linear(hidden_size, latent_dim, bias=False)

        # 5. Prediction Heads
        self.reward_head = nn.Linear(hidden_size, 1, bias=True)
        self.fail_head = nn.Linear(hidden_size, 1, bias=True)
        self.value_head = nn.Linear(hidden_size, 1, bias=True)

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            self.fail_head.bias.fill_(-0.5) # Prior bias toward viability

    def forward(
        self,
        current_state: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Dual sequence tokens: [Token 0: State, Token 1: Action] (T=2)
        tok_state = self.input_norm(self.state_proj(current_state)).unsqueeze(1)
        tok_action = self.input_norm(self.action_proj(action)).unsqueeze(1)
        h = torch.cat([tok_state, tok_action], dim=1)  # [b, 2, hidden_size]

        for layer in self.layers:
            h = layer(h)

        h_final = self.norm(h)
        h_state = h_final[:, 0, :]
        h_action = h_final[:, 1, :]

        next_state = current_state + torch.tanh(self.next_state_proj(h_state))
        reward = torch.tanh(self.reward_head(h_action).squeeze(-1))
        fail_prob = torch.sigmoid(self.fail_head(h_action).squeeze(-1))
        value = self.value_head(h_state).squeeze(-1)

        return next_state, reward, fail_prob, value

    def get_config(self) -> Dict:
        return {
            "latent_dim": self.latent_dim,
            "action_dim": self.action_dim,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "rms_norm_eps": self.rms_norm_eps,
            "rope_theta": 500000.0,
            "initializer_range": 0.02,
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "horizon_limit": self.horizon_limit,
        }

    def save_checkpoint(self, output_dir: str, max_shard_mb: int = 480):
        os.makedirs(output_dir, exist_ok=True)

        cfg_dict = self.get_config()
        cfg_path = os.path.join(output_dir, "world_model_config.json")
        alt_cfg_path = os.path.join(output_dir, "config.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg_dict, f, indent=2)
        with open(alt_cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg_dict, f, indent=2)

        tensor_dict = {
            "input_proj.state_proj.weight": self.state_proj.weight.detach().cpu().contiguous(),
            "input_proj.action_proj.weight": self.action_proj.weight.detach().cpu().contiguous(),
            "input_proj.norm.weight": self.input_norm.weight.detach().cpu().contiguous(),
            "norm.weight": self.norm.weight.detach().cpu().contiguous(),
            "heads.next_state_proj.weight": self.next_state_proj.weight.detach().cpu().contiguous(),
            "heads.reward.weight": self.reward_head.weight.detach().cpu().contiguous(),
            "heads.reward.bias": self.reward_head.bias.detach().cpu().contiguous(),
            "heads.viability_fail.weight": self.fail_head.weight.detach().cpu().contiguous(),
            "heads.viability_fail.bias": self.fail_head.bias.detach().cpu().contiguous(),
            "heads.value.weight": self.value_head.weight.detach().cpu().contiguous(),
            "heads.value.bias": self.value_head.bias.detach().cpu().contiguous(),
        }

        for l, layer in enumerate(self.layers):
            p = f"layers.{l}"
            tensor_dict[f"{p}.input_layernorm.weight"] = layer.input_layernorm.weight.detach().cpu().contiguous()
            tensor_dict[f"{p}.self_attn.q_proj.weight"] = layer.q_proj.weight.detach().cpu().contiguous()
            tensor_dict[f"{p}.self_attn.k_proj.weight"] = layer.k_proj.weight.detach().cpu().contiguous()
            tensor_dict[f"{p}.self_attn.v_proj.weight"] = layer.v_proj.weight.detach().cpu().contiguous()
            tensor_dict[f"{p}.self_attn.o_proj.weight"] = layer.o_proj.weight.detach().cpu().contiguous()
            tensor_dict[f"{p}.post_attention_layernorm.weight"] = layer.post_attention_layernorm.weight.detach().cpu().contiguous()
            tensor_dict[f"{p}.mlp.gate_proj.weight"] = layer.gate_proj.weight.detach().cpu().contiguous()
            tensor_dict[f"{p}.mlp.up_proj.weight"] = layer.up_proj.weight.detach().cpu().contiguous()
            tensor_dict[f"{p}.mlp.down_proj.weight"] = layer.down_proj.weight.detach().cpu().contiguous()

        # 1. Monolithic weights file for backwards compatibility
        monolithic_path = os.path.join(output_dir, "world_model_dynamics.safetensors")
        if HAS_SAFETENSORS:
            save_safetensors(tensor_dict, monolithic_path)
        else:
            torch.save(tensor_dict, monolithic_path.replace(".safetensors", ".pt"))

        # 2. Canonical Part-Wise Sharded SafeTensors format with model.safetensors.index.json (Rule 20)
        if HAS_SAFETENSORS:
            max_shard_bytes = max_shard_mb * 1024 * 1024
            shards: List[Dict[str, torch.Tensor]] = [{}]
            cur_shard_bytes = 0
            weight_map: Dict[str, str] = {}
            total_size = 0

            # Deterministic sorted key iteration
            for name in sorted(tensor_dict.keys()):
                tensor = tensor_dict[name]
                t_bytes = tensor.numel() * tensor.element_size()
                total_size += t_bytes

                if cur_shard_bytes > 0 and (cur_shard_bytes + t_bytes > max_shard_bytes):
                    shards.append({})
                    cur_shard_bytes = 0

                shard_idx = len(shards)
                shard_name = f"world_model-{shard_idx:05d}.safetensors"
                shards[-1][name] = tensor
                weight_map[name] = shard_name
                cur_shard_bytes += t_bytes

            num_shards = len(shards)
            for s_idx, shard_tensors in enumerate(shards, start=1):
                s_filename = f"world_model-{s_idx:05d}.safetensors"
                s_path = os.path.join(output_dir, s_filename)
                save_safetensors(shard_tensors, s_path)

            index_dict = {
                "metadata": {"total_size": total_size},
                "weight_map": weight_map,
            }
            index_path = os.path.join(output_dir, "model.safetensors.index.json")
            with open(index_path, "w", encoding="utf-8") as f:
                json.dump(index_dict, f, indent=2)


def project_tokens_to_latent(
    tokens: List[int],
    target_dim: int,
    rope_theta: float = 500000.0,
) -> torch.Tensor:
    """
    Continuous Rotary Frequency Token Projection:
    100% Mathematical Parity with Rust 'project_tokens_to_latent' in world_model.rs.
    Maps discrete token or UTF-8 byte sequences into continuous normalized latent vectors.
    """
    if not tokens:
        return torch.zeros(target_dim, dtype=torch.float32)

    half_dim = target_dim // 2
    tok_tensor = torch.tensor(tokens, dtype=torch.float32)
    pos = torch.arange(len(tokens), dtype=torch.float32)
    val_pos = tok_tensor * 0.01 + pos

    idx = torch.arange(half_dim, dtype=torch.float32)
    freq = 1.0 / (rope_theta ** ((2.0 * idx) / target_dim))

    angles = val_pos.unsqueeze(1) * freq.unsqueeze(0)
    sin_comp = torch.sin(angles).sum(dim=0)
    cos_comp = torch.cos(angles).sum(dim=0)

    vec = torch.empty(target_dim, dtype=torch.float32)
    vec[0::2] = sin_comp
    vec[1::2] = cos_comp

    norm = torch.norm(vec).item()
    if norm > 1e-6:
        vec = vec / norm
    return vec


def extract_transition_payload(rec: Dict) -> Optional[Tuple[Any, Any, Any, float, float, float]]:
    """
    Extracts authentic (state, action, next_state, reward, failed, return) from diverse record formats:
    - Pre-vectorized numerical vectors
    - Structured (state_observation, action, next_state_observation)
    - Nested 'state_transition' objects
    - Curriculum and derivation traces
    """
    if not isinstance(rec, dict):
        return None

    t_obj = rec.get("state_transition") if isinstance(rec.get("state_transition"), dict) else rec

    # 1. Direct vector or string observations
    s = t_obj.get("state") or t_obj.get("state_observation") or t_obj.get("observation")
    a = t_obj.get("action") or t_obj.get("intervention")
    ns = t_obj.get("next_state") or t_obj.get("next_state_observation") or t_obj.get("subsequent_state")

    # 2. Derivation trace extraction (from agi_causal_transitions)
    if not (s and a and ns):
        deriv = str(rec.get("derivation", ""))
        if "State observation" in deriv or "State s" in deriv:
            m = re.search(
                r"State(?:\s+observation)?:\s*(.+?)\.\s*Action:\s*(.+?)\.\s*Next(?:\s+state)?(?:\s+observation)?:\s*(.+?)(?:\.\s*Reward:|$)",
                deriv,
                re.DOTALL | re.IGNORECASE,
            )
            if m:
                s, a, ns = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()

    # 3. Fallback to concept / definition / formula transitions
    if not (s and a and ns):
        if rec.get("domain") == "agi_causal_transitions" or "transition" in str(rec.get("concept", "")).lower():
            s = str(rec.get("definition", rec.get("concept", rec.get("input", ""))))[:512].strip()
            a = str(rec.get("formula", rec.get("subject", "causal_transition")))[:256].strip()
            ns = str(rec.get("derivation", rec.get("explanation", rec.get("output", ""))))[:1024].strip()

    if not s or not a or not ns:
        return None

    # Rewards & Viability
    try:
        r = float(t_obj.get("reward", rec.get("reward", 0.0)))
    except (ValueError, TypeError):
        r = 0.0

    done_flag = bool(t_obj.get("done", rec.get("done", False)))
    failed_raw = t_obj.get("failed", rec.get("failed", t_obj.get("is_failure", rec.get("is_failure"))))
    if failed_raw is not None:
        failed = 1.0 if bool(failed_raw) else 0.0
    elif done_flag and r < 0.0:
        failed = 1.0
    else:
        failed = 0.0

    ret_raw = t_obj.get("return", rec.get("return"))
    try:
        ret = float(ret_raw) if ret_raw is not None else r
    except (ValueError, TypeError):
        ret = r

    return s, a, ns, r, failed, ret


class StateTransitionDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        latent_dim: int = 2048,
        action_dim: int = 512,
        tokenizer: Optional[Any] = None,
    ):
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.tokenizer = tokenizer
        self.records: List[Dict] = []

        candidate_dirs = [data_path]
        if os.path.isdir(data_path):
            candidate_dirs.append(os.path.join(data_path, "curriculum"))
            candidate_dirs.append(os.path.join(data_path, "transitions"))

        visited_files = set()
        for c_dir in candidate_dirs:
            if not os.path.exists(c_dir):
                continue
            if os.path.isfile(c_dir):
                if c_dir not in visited_files:
                    visited_files.add(c_dir)
                    self._load_file(c_dir)
            else:
                for root, _, files in os.walk(c_dir):
                    for f in files:
                        if f.endswith(".jsonl"):
                            fp = os.path.join(root, f)
                            if fp not in visited_files:
                                visited_files.add(fp)
                                self._load_file(fp)

        # Fail-closed validation against synthetic fallbacks or empty datasets
        if not self.records:
            raise FileNotFoundError(
                f"No authentic state transition records found at '{data_path}'. "
                "Synthetic data generation and mock fallbacks are strictly prohibited."
            )

    def _load_file(self, file_path: str):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    parsed = extract_transition_payload(obj)
                    if parsed is not None:
                        s, a, ns, r, failed, ret = parsed
                        self.records.append({
                            "s": s,
                            "a": a,
                            "ns": ns,
                            "r": r,
                            "failed": failed,
                            "ret": ret,
                        })
                except json.JSONDecodeError:
                    continue

    def __len__(self) -> int:
        return len(self.records)

    def _encode_feature(self, val: Any, target_dim: int) -> torch.Tensor:
        if isinstance(val, (list, tuple)):
            if len(val) == target_dim:
                return torch.tensor(val, dtype=torch.float32)
            elif len(val) > 0 and all(isinstance(x, (int, float)) for x in val):
                t = torch.tensor(val, dtype=torch.float32)
                if len(t) < target_dim:
                    return F.pad(t, (0, target_dim - len(t)))
                else:
                    return t[:target_dim]
        elif isinstance(val, torch.Tensor):
            if val.numel() == target_dim:
                return val.view(-1).float()

        # String observation encoding into continuous latent vector
        text = str(val)
        if self.tokenizer is not None and hasattr(self.tokenizer, "encode"):
            try:
                res = self.tokenizer.encode(text)
                tokens = res.ids if hasattr(res, "ids") else list(res)
            except Exception:
                tokens = [b for b in text.encode("utf-8")]
        else:
            tokens = [b for b in text.encode("utf-8")]

        return project_tokens_to_latent(tokens, target_dim)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx]
        if "s_tensor" not in rec:
            rec["s_tensor"] = self._encode_feature(rec["s"], self.latent_dim)
            rec["a_tensor"] = self._encode_feature(rec["a"], self.action_dim)
            rec["ns_tensor"] = self._encode_feature(rec["ns"], self.latent_dim)
            rec["r_tensor"] = torch.tensor(rec["r"], dtype=torch.float32)
            rec["failed_tensor"] = torch.tensor(rec["failed"], dtype=torch.float32)
            rec["ret_tensor"] = torch.tensor(rec["ret"], dtype=torch.float32)

        return {
            "state": rec["s_tensor"],
            "action": rec["a_tensor"],
            "next_state": rec["ns_tensor"],
            "reward": rec["r_tensor"],
            "failed": rec["failed_tensor"],
            "return": rec["ret_tensor"],
        }


def train_world_model_gpu(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[World Model GPU Training] Device: {device}")

    model = NeuralWorldModel(
        latent_dim=args.latent_dim,
        action_dim=args.action_dim,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.layers,
        num_attention_heads=args.heads,
        num_key_value_heads=args.kv_heads,
        head_dim=args.head_dim,
        learning_rate=args.lr,
    ).to(device)

    dataset = StateTransitionDataset(args.data_path, latent_dim=args.latent_dim, action_dim=args.action_dim)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Multi-task loss balancing weights: prioritize state dynamics learning
    lambda_dyn = args.lambda_dyn
    lambda_rew = args.lambda_rew
    lambda_fail = args.lambda_fail
    lambda_val = args.lambda_val

    print(f"[World Model GPU] Loaded {len(dataset)} records. Loss weights: dyn={lambda_dyn}, rew={lambda_rew}, fail={lambda_fail}, val={lambda_val}")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_dyn = 0.0
        total_rew = 0.0
        total_fail = 0.0
        total_val = 0.0

        for batch in dataloader:
            s = batch["state"].to(device)
            a = batch["action"].to(device)
            ns = batch["next_state"].to(device)
            r = batch["reward"].to(device)
            failed = batch["failed"].to(device)
            ret = batch["return"].to(device)

            pred_ns, pred_r, pred_fail, pred_val = model(s, a)

            loss_dyn = F.mse_loss(pred_ns, ns)
            loss_rew = F.mse_loss(pred_r, r)
            loss_fail = F.binary_cross_entropy(pred_fail, failed)
            loss_val = F.mse_loss(pred_val, ret)

            loss = (
                lambda_dyn * loss_dyn
                + lambda_rew * loss_rew
                + lambda_fail * loss_fail
                + lambda_val * loss_val
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_dyn += loss_dyn.item()
            total_rew += loss_rew.item()
            total_fail += loss_fail.item()
            total_val += loss_val.item()

        n = max(len(dataloader), 1)
        print(
            f"  Epoch [{epoch+1}/{args.epochs}] Total Loss: {total_loss/n:.4f} | "
            f"Dynamics MSE: {total_dyn/n:.4f} | Reward MSE: {total_rew/n:.4f} | Fail BCE: {total_fail/n:.4f}"
        )

    # Export trained weights
    model.save_checkpoint(args.output_dir)

    # Export numerical parity probe for Rust cross-verification
    model.eval()
    with torch.no_grad():
        sample_s = torch.ones(1, args.latent_dim, device=device) * 0.1
        sample_a = torch.ones(1, args.action_dim, device=device) * 0.05
        out_ns, out_r, out_fail, out_val = model(sample_s, sample_a)

        parity_probe = {
            "input_state": sample_s[0].cpu().tolist(),
            "input_action": sample_a[0].cpu().tolist(),
            "expected_next_state": out_ns[0].cpu().tolist(),
            "expected_reward": float(out_r[0].cpu().item()),
            "expected_failure_probability": float(out_fail[0].cpu().item()),
            "expected_value": float(out_val[0].cpu().item()),
        }
        with open(os.path.join(args.output_dir, "parity_evaluation.json"), "w", encoding="utf-8") as pf:
            json.dump(parity_probe, pf, indent=2)

    print(f"[World Model GPU] Exported checkpoint and parity evaluation to {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="World Model GPU Trainer")
    parser.add_argument("--data-path", type=str, default="storage/datasets/world_model")
    parser.add_argument("--output-dir", type=str, default="storage/models/tara_world_model")
    parser.add_argument("--latent-dim", type=int, default=2048)
    parser.add_argument("--action-dim", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--intermediate-size", type=int, default=5632)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--lambda-dyn", type=float, default=2.0)
    parser.add_argument("--lambda-rew", type=float, default=1.0)
    parser.add_argument("--lambda-fail", type=float, default=1.0)
    parser.add_argument("--lambda-val", type=float, default=0.5)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    train_world_model_gpu(args)


if __name__ == "__main__":
    main()
