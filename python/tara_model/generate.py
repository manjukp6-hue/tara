"""
python/tara_model/generate.py

Real Autoregressive Language Generation & Evaluation for TARA.
Directly executes forward passes through model.safetensors weights.
Zero hardcoded or mocked replies.
"""

import os
import sys
import json
import time
import math
import struct
import random
from typing import List, Dict, Any, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tara_model.tokenizer import TaraTokenizer
from tara_model.architecture import TaraModelZero, TaraConfig

def unpack_safetensors_tensor(dtype_str: str, shape: List[int], raw_bytes: bytes) -> List[Any]:
    """
    Unpacks raw SafeTensors binary data according to the exact dtype metadata.
    Preserves exact IEEE 754 precision, shape, and endianness without lossy conversion.
    Supports F32, F64, F16, BF16, I32, I64, I16, I8, U8, and BOOL.
    """
    flat_count = math.prod(shape)
    dtype_map = {
        "F32": ("<f", 4),
        "F64": ("<d", 8),
        "F16": ("<e", 2),
        "I32": ("<i", 4),
        "I64": ("<q", 8),
        "I16": ("<h", 2),
        "I8":  ("<b", 1),
        "U8":  ("<B", 1),
        "BOOL":("<?", 1),
    }

    if dtype_str == "BF16":
        expected_bytes = flat_count * 2
        if len(raw_bytes) < expected_bytes:
            raise ValueError(f"Buffer underflow: expected {expected_bytes} bytes for {flat_count} BF16, got {len(raw_bytes)}")
        raw_u16 = struct.unpack_from(f"<{flat_count}H", raw_bytes, 0)
        # BF16 represents the upper 16 bits of a 32-bit IEEE 754 float
        floats = [struct.unpack("<f", struct.pack("<I", v << 16))[0] for v in raw_u16]
        return floats

    if dtype_str not in dtype_map:
        raise ValueError(f"Unsupported SafeTensors dtype: '{dtype_str}'")

    fmt_char, elem_size = dtype_map[dtype_str]
    expected_bytes = flat_count * elem_size
    if len(raw_bytes) < expected_bytes:
        raise ValueError(f"Buffer underflow: expected {expected_bytes} bytes for {flat_count} elements of {dtype_str}, got {len(raw_bytes)}")

    vals = list(struct.unpack_from(f"<{flat_count}{fmt_char[1]}", raw_bytes, 0))
    return vals

def load_trained_language_model(model_dir="storage/models/tara"):
    if not os.path.exists(model_dir):
        legacy_dir = "storage/models/tara-language"
        if os.path.exists(legacy_dir):
            model_dir = legacy_dir
    config_path = os.path.join(model_dir, "config.json")
    weights_path = os.path.join(model_dir, "model.safetensors")
    index_path = os.path.join(model_dir, "model.safetensors.index.json")

    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)

    config = TaraConfig(
        vocab_size=config_dict.get("vocab_size", 344),
        hidden_size=config_dict.get("hidden_size", 64),
        intermediate_size=config_dict.get("intermediate_size", 128),
        num_hidden_layers=config_dict.get("num_hidden_layers", 2),
        num_attention_heads=config_dict.get("num_attention_heads", 4),
        num_key_value_heads=config_dict.get("num_key_value_heads", 2),
        version=config_dict.get("version", "TARA")
    )

    model = TaraModelZero(config)

    # Helper to unpack and load from raw safetensors data buffer
    def load_tensors_from_safetensors_bytes(raw_bytes: bytes, target_weights: Dict[str, Any]):
        header_size = int.from_bytes(raw_bytes[:8], "little")
        header_json = json.loads(raw_bytes[8:8+header_size].decode("utf-8"))
        for tname, tinfo in header_json.items():
            if tname == "__metadata__":
                continue
            offsets = tinfo["data_offsets"]
            t_bytes = raw_bytes[8+header_size+offsets[0] : 8+header_size+offsets[1]]
            shape = tinfo["shape"]
            dtype_str = tinfo.get("dtype", "F32")

            floats = unpack_safetensors_tensor(dtype_str, shape, t_bytes)

            if len(shape) == 2:
                rows, cols = shape
                t_matrix = []
                for r in range(rows):
                    t_matrix.append(floats[r*cols : (r+1)*cols])
                target_weights[tname] = t_matrix
            else:
                target_weights[tname] = floats

    # Case A: Sharded SafeTensors (model.safetensors.index.json present)
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            idx_data = json.load(f)
        weight_map = idx_data.get("weight_map", {})
        unique_shards = sorted(list(set(weight_map.values())))
        for shard_name in unique_shards:
            shard_file = os.path.join(model_dir, shard_name)
            if not os.path.exists(shard_file):
                raise FileNotFoundError(f"Missing SafeTensors shard: {shard_file}")
            with open(shard_file, "rb") as sf:
                s_data = sf.read()
            load_tensors_from_safetensors_bytes(s_data, model.weights)

    # Case B: Single SafeTensors file
    elif os.path.exists(weights_path):
        with open(weights_path, "rb") as f:
            data = f.read()
        load_tensors_from_safetensors_bytes(data, model.weights)
    else:
        raise FileNotFoundError(f"No SafeTensors weights found in {model_dir}")

    # Compute parameter count dynamically from loaded tensors
    total_params = 0
    for tensor in model.weights.values():
        if isinstance(tensor, list):
            if tensor and isinstance(tensor[0], list):
                total_params += len(tensor) * len(tensor[0])
            else:
                total_params += len(tensor)
    config_dict["total_parameters"] = total_params

    tokenizer = TaraTokenizer(vocab_size=config.vocab_size)
    return model, tokenizer, config_dict

def sample_next_token(
    logits: List[float],
    generated_ids: Optional[List[int]] = None,
    temperature: float = 0.7,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0
) -> int:
    """
    Applies repetition penalty, temperature scaling, top-k filtering,
    and top-p (nucleus sampling) to select the next token id.
    """
    logits = list(logits)
    vocab_size = len(logits)

    # 1. Repetition penalty (Keskar et al. 2019)
    if repetition_penalty != 1.0 and generated_ids:
        seen = set(generated_ids)
        for tid in seen:
            if 0 <= tid < vocab_size:
                if logits[tid] < 0:
                    logits[tid] *= repetition_penalty
                else:
                    logits[tid] /= repetition_penalty

    # 2. Temperature scaling
    if temperature <= 0.001:
        return logits.index(max(logits))

    m_val = max(logits)
    scaled = [(v - m_val) / max(0.01, temperature) for v in logits]
    exps = [math.exp(max(-30.0, min(30.0, s))) for s in scaled]
    sum_exps = sum(exps)
    probs = [e / max(1e-12, sum_exps) for e in exps]

    indexed_probs = list(enumerate(probs))

    # 3. Top-K filtering
    if 0 < top_k < vocab_size:
        indexed_probs.sort(key=lambda x: x[1], reverse=True)
        indexed_probs = indexed_probs[:top_k]
        sub_sum = sum(p for _, p in indexed_probs)
        indexed_probs = [(idx, p / max(1e-12, sub_sum)) for idx, p in indexed_probs]

    # 4. Top-P (Nucleus) filtering
    if 0.0 < top_p < 1.0:
        indexed_probs.sort(key=lambda x: x[1], reverse=True)
        cum_prob = 0.0
        cutoff_idx = len(indexed_probs)
        for i, (_, p) in enumerate(indexed_probs):
            cum_prob += p
            if cum_prob >= top_p:
                cutoff_idx = i + 1
                break
        indexed_probs = indexed_probs[:cutoff_idx]
        sub_sum = sum(p for _, p in indexed_probs)
        indexed_probs = [(idx, p / max(1e-12, sub_sum)) for idx, p in indexed_probs]

    if len(indexed_probs) == 1 or temperature < 0.05:
        return max(indexed_probs, key=lambda x: x[1])[0]

    # Sample from distribution
    r = random.random()
    acc = 0.0
    for idx, p in indexed_probs:
        acc += p
        if r <= acc:
            return idx
    return indexed_probs[-1][0]


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 25,
    temperature: float = 0.7,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    stop_tokens: Optional[List[str]] = None
) -> Dict[str, Any]:
    input_tokens = tokenizer.encode(prompt)
    if not input_tokens:
        input_tokens = [tokenizer.token_to_id.get("<|im_start|>", 1)]

    curr_tokens = list(input_tokens)
    generated_ids = []

    stop_ids = {
        tokenizer.token_to_id.get("<|im_end|>", 2),
        tokenizer.token_to_id.get("<|pad|>", 0)
    }
    if stop_tokens:
        for st in stop_tokens:
            if st in tokenizer.token_to_id:
                stop_ids.add(tokenizer.token_to_id[st])

    t_start = time.perf_counter()
    t_first = None

    for step in range(max_new_tokens):
        logits, _ = model.forward(curr_tokens)
        if t_first is None:
            t_first = time.perf_counter()

        last_logits = logits[-1]
        best_id = sample_next_token(
            last_logits,
            generated_ids=generated_ids,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty
        )

        if best_id in stop_ids:
            break

        generated_ids.append(best_id)
        curr_tokens.append(best_id)

    t_end = time.perf_counter()
    first_latency_ms = (t_first - t_start) * 1000 if t_first else 0.0
    total_latency_ms = (t_end - t_start) * 1000
    tps = len(generated_ids) / max(0.0001, (t_end - t_start))

    generated_text = tokenizer.decode(generated_ids)
    return {
        "text": generated_text,
        "token_count": len(generated_ids),
        "first_latency_ms": round(first_latency_ms, 2),
        "total_latency_ms": round(total_latency_ms, 2),
        "tps": round(tps, 2)
    }


def generate_stream(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.7,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    stop_tokens: Optional[List[str]] = None
):
    """
    Generator yielding token strings in real-time as they are inferred.
    """
    input_tokens = tokenizer.encode(prompt)
    if not input_tokens:
        input_tokens = [tokenizer.token_to_id.get("<|im_start|>", 1)]

    curr_tokens = list(input_tokens)
    generated_ids = []

    stop_ids = {
        tokenizer.token_to_id.get("<|im_end|>", 2),
        tokenizer.token_to_id.get("<|pad|>", 0)
    }
    if stop_tokens:
        for st in stop_tokens:
            if st in tokenizer.token_to_id:
                stop_ids.add(tokenizer.token_to_id[st])

    for step in range(max_new_tokens):
        logits, _ = model.forward(curr_tokens)
        last_logits = logits[-1]

        best_id = sample_next_token(
            last_logits,
            generated_ids=generated_ids,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty
        )

        if best_id in stop_ids:
            break

        generated_ids.append(best_id)
        curr_tokens.append(best_id)

        token_str = tokenizer.decode([best_id])
        yield token_str


class ControlTokenAction:
    def __init__(self, action_type: str, target: str, payload: Dict[str, Any], raw_text: str):
        self.action_type = action_type  # TOOL, SKILL, RULE, CREATOR_AUTH
        self.target = target
        self.payload = payload
        self.raw_text = raw_text

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_type": self.action_type,
            "target": self.target,
            "payload": self.payload,
            "raw_text": self.raw_text
        }


class ControlTokenActionParser:
    """
    Parses structural control token blocks emitted by neural model:
    - <|tara_exec|> {"tool": "file_inspector", "params": {...}} <|im_end|>
    - <|tara_skill|> {"skill": "diagnostics", "params": {...}} <|im_end|>
    - <|creator_auth|> ...
    - <|tara_rule|> ...
    """
    @classmethod
    def parse(cls, output_text: str) -> Optional[ControlTokenAction]:
        if not output_text or not isinstance(output_text, str):
            return None

        # 1. Tool execution tag
        if "<|tara_exec|>" in output_text:
            part = output_text.split("<|tara_exec|>", 1)[1]
            part = part.split("<|im_end|>", 1)[0].strip()
            try:
                data = json.loads(part)
                tool_name = data.get("tool") or data.get("name", "generic_tool")
                params = data.get("params") or data.get("args") or {}
                return ControlTokenAction(
                    action_type="TOOL",
                    target=tool_name,
                    payload=params,
                    raw_text=output_text
                )
            except Exception:
                # Basic string format fallback: tool_name param1=val1
                tokens = part.split()
                if tokens:
                    return ControlTokenAction(
                        action_type="TOOL",
                        target=tokens[0],
                        payload={"args": tokens[1:]},
                        raw_text=output_text
                    )

        # 2. Skill execution tag
        if "<|tara_skill|>" in output_text:
            part = output_text.split("<|tara_skill|>", 1)[1]
            part = part.split("<|im_end|>", 1)[0].strip()
            try:
                data = json.loads(part)
                skill_name = data.get("skill") or data.get("name", "generic_skill")
                params = data.get("params") or data.get("args") or {}
                return ControlTokenAction(
                    action_type="SKILL",
                    target=skill_name,
                    payload=params,
                    raw_text=output_text
                )
            except Exception:
                tokens = part.split()
                if tokens:
                    return ControlTokenAction(
                        action_type="SKILL",
                        target=tokens[0],
                        payload={"args": tokens[1:]},
                        raw_text=output_text
                    )

        # 3. Creator authorization tag
        if "<|creator_auth|>" in output_text:
            return ControlTokenAction(
                action_type="CREATOR_AUTH",
                target="creator_confirmation",
                payload={"statement": output_text.replace("<|creator_auth|>", "").strip()},
                raw_text=output_text
            )

        # 4. Rule tag
        if "<|tara_rule|>" in output_text:
            return ControlTokenAction(
                action_type="RULE",
                target="rule_evaluation",
                payload={"statement": output_text.replace("<|tara_rule|>", "").strip()},
                raw_text=output_text
            )

        return None

def run_evaluation_suite(model_dir="storage/models/tara-language"):
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    
    model, tokenizer, config = load_trained_language_model(model_dir)
    print("=" * 68)
    print("      TARA NEURAL INFERENCE BENCHMARK (REAL LANGUAGE GENERATION)")
    print("=" * 68 + "\n")
    
    prompts = [
        ("1. Identity", "Hello, who are you?"),
        ("2. Arithmetic", "What is 2 + 2?"),
        ("3. TARA Architecture", "Explain what TARA is."),
        ("4. Coding (Python)", "Write a short Python function that adds two numbers."),
        ("5. Kannada Language", "Respond in Kannada: TARA yenu?"),
        ("6. Technical Reasoning", "What is the mathematical definition of grouped query attention?")
    ]
    
    results = []
    for cat, p in prompts:
        gen = generate_response(model, tokenizer, p, max_new_tokens=30, temperature=0.6)
        results.append({
            "category": cat,
            "prompt": p,
            "response": gen["text"],
            "tokens": gen["token_count"],
            "first_latency_ms": gen["first_latency_ms"],
            "total_latency_ms": gen["total_latency_ms"],
            "tps": gen["tps"]
        })
        print(f"[{cat}] Prompt: \"{p}\"")
        print(f"       Generated Output: {gen['text']}")
        print(f"       Metrics: {gen['tokens']} tokens | Latency: {gen['total_latency_ms']} ms | Speed: {gen['tps']} tok/s\n")
        
    return results

if __name__ == "__main__":
    run_evaluation_suite()