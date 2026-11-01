#!/usr/bin/env python3
"""
TARA Standalone Checkpoint & Architecture Verification Engine
=============================================================================
Verifies that an exported model checkpoint (model.safetensors, config.json,
tokenizer.json) is 100% compliant with TARA's native Rust tara_engine.

Usage:
    python verify_model.py --model_dir path/to/checkpoint
=============================================================================
"""

import argparse
import json
import os
import sys
import torch

try:
    import safetensors
    from safetensors.torch import load_file
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

from model_dataset import TaraForCausalLM, TaraNativeTokenizer
from gdrive_sync import load_model_weights_dir


def verify_checkpoint(model_dir: str) -> bool:
    print("=" * 75)
    print(f"  VERIFYING TARA MODEL COMPATIBILITY: {model_dir}")
    print("=" * 75)

    if not os.path.exists(model_dir):
        print(f"[FAIL] Directory does not exist: {model_dir}")
        return False

    safetensors_path = os.path.join(model_dir, "model.safetensors")
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    config_path = os.path.join(model_dir, "config.json")
    tokenizer_path = os.path.join(model_dir, "tokenizer.json")

    has_single_st = os.path.isfile(safetensors_path) and os.path.getsize(safetensors_path) > 0
    has_sharded_st = False
    sharded_files = []
    if os.path.isfile(index_path) and os.path.getsize(index_path) > 0:
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                idx_data = json.load(f)
            wmap = idx_data.get("weight_map", {})
            if wmap:
                sharded_files = sorted(set(wmap.values()))
                if sharded_files and all(os.path.isfile(os.path.join(model_dir, sf)) and os.path.getsize(os.path.join(model_dir, sf)) > 0 for sf in sharded_files):
                    has_sharded_st = True
        except Exception:
            has_sharded_st = False

    # 1. Check file existence
    if not (has_single_st or has_sharded_st):
        print(f"[FAIL] Missing valid model weights (neither model.safetensors nor verified sharded index/shards found): {model_dir}")
        return False
    if not os.path.exists(config_path):
        print(f"[FAIL] config.json missing in {model_dir}")
        return False

    cfg_size = os.path.getsize(config_path)
    if has_single_st:
        sf_size = os.path.getsize(safetensors_path)
        print(f"[1/5] Files found: model.safetensors ({sf_size:,} bytes), config.json ({cfg_size:,} bytes)")
    else:
        idx_size = os.path.getsize(index_path)
        print(f"[1/5] Files found: model.safetensors.index.json ({idx_size:,} bytes, {len(sharded_files)} shards), config.json ({cfg_size:,} bytes)")
    if os.path.exists(tokenizer_path):
        print(f"      tokenizer.json found ({os.path.getsize(tokenizer_path):,} bytes)")

    # 2. Check config.json keys against TaraConfig in tara_engine
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    required_keys = [
        "vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
        "num_attention_heads", "num_key_value_heads", "head_dim",
        "max_position_embeddings", "rope_theta", "rms_norm_eps"
    ]
    missing = [k for k in required_keys if k not in config]
    if missing:
        print(f"[FAIL] Missing required TaraConfig keys in config.json: {missing}")
        return False
    print("[2/5] config.json validated: All required architectural keys present!")

    # 3. Check SafeTensors weight keys
    if not HAS_SAFETENSORS:
        print("[FAIL] safetensors package is not installed.")
        return False

    keys = set()
    if has_single_st:
        with safetensors.safe_open(safetensors_path, framework="pt", device="cpu") as sf:
            keys.update(sf.keys())
    elif has_sharded_st:
        for sf_name in sharded_files:
            sf_path = os.path.join(model_dir, sf_name)
            with safetensors.safe_open(sf_path, framework="pt", device="cpu") as sf:
                keys.update(sf.keys())

    num_layers = config["num_hidden_layers"]
    expected_keys = [
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
    ]
    for layer_idx in range(num_layers):
        expected_keys.extend([
            f"model.layers.{layer_idx}.input_layernorm.weight",
            f"model.layers.{layer_idx}.self_attn.q_proj.weight",
            f"model.layers.{layer_idx}.self_attn.k_proj.weight",
            f"model.layers.{layer_idx}.self_attn.v_proj.weight",
            f"model.layers.{layer_idx}.self_attn.o_proj.weight",
            f"model.layers.{layer_idx}.post_attention_layernorm.weight",
            f"model.layers.{layer_idx}.mlp.gate_proj.weight",
            f"model.layers.{layer_idx}.mlp.up_proj.weight",
            f"model.layers.{layer_idx}.mlp.down_proj.weight",
        ])

    missing_weights = [k for k in expected_keys if k not in keys]
    if missing_weights:
        print(f"[FAIL] Missing {len(missing_weights)} required model weights: {missing_weights[:5]}...")
        return False
    print(f"[3/5] Safetensors tensor dictionary validated ({len(keys)} tensors, all required layers present)!")

    # 4. Load weights into TaraForCausalLM
    print("[4/5] Instantiating PyTorch model and loading safetensors state_dict...")
    model = TaraForCausalLM(config)
    loaded = load_model_weights_dir(model, model_dir, prefix="model")
    if not loaded:
        print(f"[FAIL] Failed to load model weights from {model_dir}")
        return False
    model.eval()
    print("      Model instantiated and weights loaded cleanly!")

    # 5. Execute forward pass test
    print("[5/5] Executing forward pass test inference...")
    test_seq_len = min(32, config.get("max_position_embeddings", 512))
    dummy_input = torch.randint(0, config["vocab_size"], (1, test_seq_len))
    with torch.no_grad():
        output = model(input_ids=dummy_input)
    
    if output.logits is None:
        print("[FAIL] Output logits are None")
        return False
    
    logits_shape = output.logits.shape
    expected_shape = (1, test_seq_len, config["vocab_size"])
    if logits_shape != expected_shape:
        print(f"[FAIL] Logits shape mismatch: expected {expected_shape}, got {logits_shape}")
        return False

    print(f"      Forward pass completed successfully! Output shape: {logits_shape}")
    print("\n" + "=" * 75)
    print("  VERIFICATION RESULT: 100% PASS - READY FOR NATIVE RUST tara_engine!")
    print("=" * 75 + "\n")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify TARA checkpoint compatibility.")
    parser.add_argument("--model_dir", type=str, default=".", help="Path to checkpoint directory containing model.safetensors and config.json")
    args = parser.parse_args()

    success = verify_checkpoint(args.model_dir)
    sys.exit(0 if success else 1)
