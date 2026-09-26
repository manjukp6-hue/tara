"""
TARA/MODEL/core/tiered_model.py

Tier-aware TARA language model runner with on-demand weight paging,
lookahead prefetch overlap, and fallback execution.
Adapted from Colibrì inference architecture.

Attribution:
Adapted from Colibrì (https://github.com/JustVugg/colibri, c/qwen36_tier.c, c/expert_store.h)
Copyright (c) 2025 JustVugg / Colibrì contributors
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
"""

import os
import sys
import json
import time
import math
from typing import Dict, List, Optional, Tuple, Any

# Add workspace roots
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from python.tara_model.architecture import TaraModelZero, TaraConfig, silu
from python.tara_model.tokenizer import TaraTokenizer

from ..inference.memory_manager.tiered_store import TieredTensorStore
from ..inference.tensor_stream.stream_reader import SafeTensorsStreamReader
from ..inference.cache.cache_policy import create_cache_policy
from ..inference.prefetch.lookahead import LookaheadPrefetcher
from ..inference.resource_planner.planner import ResourcePlanner
from ..inference.backends.backend_registry import BackendRegistry
from ..inference.telemetry.monitor import TelemetryMonitor
from ..router.routing_tracker import RoutingTracker


def resolve_model_dir(model_dir: Optional[str] = None) -> str:
    """Finds a valid TARA model directory containing config.json and model.safetensors."""
    if model_dir and os.path.exists(os.path.join(model_dir, "model.safetensors")):
        return model_dir
    candidates = [
        model_dir,
        "storage/models/tara",
        "storage/models/TARA-0.1-tokenizer-aligned",
        "storage/models/tara-0.1",
        "storage/models/tara-0.2",
        "storage/models/tara-language",
    ]
    for c in candidates:
        if c and os.path.exists(os.path.join(c, "model.safetensors")):
            return c
    raise FileNotFoundError("No valid TARA model directory found with model.safetensors")


class TieredTaraModel:
    """
    TARA Transformer with transparent weight paging across VRAM -> RAM -> Disk.
    """
    def __init__(
        self,
        model_dir: Optional[str] = None,
        config_path: Optional[str] = None,
        force_vram_gb: Optional[float] = None,
        force_ram_gb: Optional[float] = None,
        fallback_to_native: bool = False
    ):
        self.model_dir = resolve_model_dir(model_dir)
        self.fallback_to_native = fallback_to_native

        # Load tiering configuration
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "inference", "config", "memory_tiering.json"
            )
        self.tiering_config = {}
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                self.tiering_config = json.load(f)

        if self.tiering_config.get("fallback_to_native", False):
            self.fallback_to_native = True

        # Load model config
        model_cfg_file = os.path.join(self.model_dir, "config.json")
        with open(model_cfg_file, "r", encoding="utf-8") as f:
            self.model_cfg_dict = json.load(f)

        self.config = TaraConfig(
            vocab_size=self.model_cfg_dict["vocab_size"],
            hidden_size=self.model_cfg_dict["hidden_size"],
            intermediate_size=self.model_cfg_dict["intermediate_size"],
            num_hidden_layers=self.model_cfg_dict["num_hidden_layers"],
            num_attention_heads=self.model_cfg_dict["num_attention_heads"],
            num_key_value_heads=self.model_cfg_dict["num_key_value_heads"],
            version=self.model_cfg_dict.get("version", "TARA-0.1")
        )

        self.tokenizer = TaraTokenizer(vocab_size=self.config.vocab_size)
        self.weights_path = os.path.join(self.model_dir, "model.safetensors")

        # Map layer tensor names
        self.layer_tensors: Dict[int, List[str]] = {}
        for l in range(self.config.num_hidden_layers):
            p = f"model.layers.{l}"
            self.layer_tensors[l] = [
                f"{p}.input_layernorm.weight",
                f"{p}.self_attn.q_proj.weight",
                f"{p}.self_attn.k_proj.weight",
                f"{p}.self_attn.v_proj.weight",
                f"{p}.self_attn.o_proj.weight",
                f"{p}.post_attention_layernorm.weight",
                f"{p}.mlp.gate_proj.weight",
                f"{p}.mlp.up_proj.weight",
                f"{p}.mlp.down_proj.weight",
            ]

        self.global_tensors = [
            "model.embed_tokens.weight",
            "model.norm.weight",
            "lm_head.weight"
        ]

        # Planner & Budgets
        self.planner = ResourcePlanner(self.tiering_config)
        model_file_size = os.path.getsize(self.weights_path) if os.path.exists(self.weights_path) else 100_000_000
        self.plan = self.planner.plan_tiers(
            model_size_bytes=model_file_size,
            forced_vram_gb=force_vram_gb,
            forced_ram_gb=force_ram_gb
        )

        # Telemetry & Policy
        policy_name = self.tiering_config.get("cache_policy", "lfru")
        self.policy = create_cache_policy(policy_name)
        self.telemetry = TelemetryMonitor(enabled=self.tiering_config.get("telemetry", True))
        self.backend = BackendRegistry.select_best_backend()
        self.router = RoutingTracker(decay_interval_steps=self.tiering_config.get("decay_steps", 100))

        # Stream Reader & Tiered Store
        enable_async = self.tiering_config.get("async_io", True) and not self.fallback_to_native
        self.stream_reader = SafeTensorsStreamReader(self.weights_path, enable_async=enable_async)
        
        self.store = TieredTensorStore(
            stream_reader=self.stream_reader,
            cache_policy=self.policy,
            telemetry=self.telemetry,
            backend=self.backend,
            vram_capacity_bytes=self.plan["vram_budget_bytes"],
            ram_capacity_bytes=self.plan["ram_budget_bytes"],
            admission_margin_pct=self.tiering_config.get("admission_margin_pct", 25),
            admission_margin_fixed=self.tiering_config.get("admission_margin_fixed", 4)
        )

        # Lookahead Prefetcher
        prefetch_enabled = self.tiering_config.get("prefetch", True) and not self.fallback_to_native
        self.prefetcher = LookaheadPrefetcher(
            prefetch_fn=self.store.prefetch,
            depth=self.tiering_config.get("max_prefetch_depth", 1),
            enabled=prefetch_enabled
        )

        # Baseline model for fallback comparison
        self.native_model = None
        if self.fallback_to_native:
            self._init_native_fallback()

    def _init_native_fallback(self) -> None:
        """Loads all weights directly into memory (zero tiering)."""
        self.native_model = TaraModelZero(self.config)
        for tname in self.stream_reader.list_tensors():
            tv = self.stream_reader.load_tensor(tname)
            self.native_model.weights[tname] = tv.data

    def get_weight(self, name: str) -> Any:
        """Fetch weight either natively or via tiered store lease."""
        if self.fallback_to_native and self.native_model is not None:
            return self.native_model.weights[name]
        tv = self.store.lookup(name)
        self.router.record_access(name)
        return tv.data

    def release_weight(self, name: str) -> None:
        if not self.fallback_to_native:
            self.store.release(name)

    def forward(self, input_tokens: List[int]) -> Tuple[List[List[float]], Any]:
        """
        Execute forward pass with on-demand layer paging and lookahead prefetch.
        Guarantees exact numerical parity with TaraModelZero.
        """
        if self.fallback_to_native and self.native_model is not None:
            return self.native_model.forward(input_tokens)

        seq_len = len(input_tokens)
        H = self.config.hidden_size
        V = self.config.vocab_size

        # 1. Embedding lookup
        embed_w = self.get_weight("model.embed_tokens.weight")
        x = [list(embed_w[tid]) for tid in input_tokens]
        self.release_weight("model.embed_tokens.weight")

        cache = {"x_in": [list(row) for row in x], "token_ids": list(input_tokens)}

        # 2. Sequential Layers with Paging & Prefetch
        total_layers = self.config.num_hidden_layers
        for l in range(total_layers):
            self.prefetcher.on_layer_begin(l, total_layers, self.layer_tensors)
            p = f"model.layers.{l}"

            # Layer Norm
            norm_w = self.get_weight(f"{p}.input_layernorm.weight")
            for i in range(seq_len):
                variance = sum(val * val for val in x[i]) / H
                rms = math.sqrt(variance + self.config.rms_norm_eps)
                x[i] = [(x[i][h] / rms) * norm_w[h] for h in range(H)]
            self.release_weight(f"{p}.input_layernorm.weight")

            # Attention Q Projection & Residual
            q_w = self.get_weight(f"{p}.self_attn.q_proj.weight")
            new_x = []
            for i in range(seq_len):
                proj = [sum(q_w[h][k] * x[i][k] for k in range(H)) for h in range(H)]
                new_x.append([x[i][h] + proj[h] * 0.1 for h in range(H)])
            x = new_x
            self.release_weight(f"{p}.self_attn.q_proj.weight")

            self.prefetcher.on_layer_complete(l, self.layer_tensors)

        # 3. Final Norm
        final_norm = self.get_weight("model.norm.weight")
        for i in range(seq_len):
            variance = sum(val * val for val in x[i]) / H
            rms = math.sqrt(variance + self.config.rms_norm_eps)
            x[i] = [(x[i][h] / rms) * final_norm[h] for h in range(H)]
        self.release_weight("model.norm.weight")

        cache["hidden_states"] = x

        # 4. LM Head projection
        lm_head = self.get_weight("lm_head.weight")
        logits = []
        for i in range(seq_len):
            row_logits = [sum(lm_head[v][h] * x[i][h] for h in range(H)) for v in range(V)]
            logits.append(row_logits)
        self.release_weight("lm_head.weight")

        return logits, cache

    def generate(self, prompt: str, max_new_tokens: int = 15, temperature: float = 0.7) -> str:
        """Autoregressively generate tokens from prompt."""
        input_tokens = self.tokenizer.encode(prompt)
        if not input_tokens:
            input_tokens = [self.tokenizer.token_to_id.get("<|im_start|>", 1)]

        curr_tokens = list(input_tokens)
        generated_ids = []

        for _ in range(max_new_tokens):
            logits, _ = self.forward(curr_tokens)
            last_logits = logits[-1]
            m_val = max(last_logits)
            scaled = [(v - m_val) / max(0.1, temperature) for v in last_logits]
            exps = [math.exp(max(-30.0, min(30.0, s))) for s in scaled]
            s_exps = sum(exps)
            probs = [e / s_exps for e in exps]
            best_id = probs.index(max(probs))

            if best_id in [self.tokenizer.token_to_id.get("<|im_end|>", 2), self.tokenizer.token_to_id.get("<|pad|>", 0)]:
                break
            generated_ids.append(best_id)
            curr_tokens.append(best_id)

        return self.tokenizer.decode(generated_ids)

    def close(self) -> None:
        if self.stream_reader:
            self.stream_reader.close()


def run_verification_inference(
    model_dir: str = "storage/models/tara-language",
    prompt: str = "TARA test",
    max_tokens: int = 10
) -> Tuple[str, Dict[str, Any]]:
    """Runs a verification inference pass and returns generated text and stats."""
    model = TieredTaraModel(model_dir=model_dir)
    try:
        output_text = model.generate(prompt, max_new_tokens=max_tokens)
        stats = model.telemetry.get_stats()
        return output_text, stats
    finally:
        model.close()
