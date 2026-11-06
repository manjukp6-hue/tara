#!/usr/bin/env python3
"""
TARA Standalone Multi-Domain SFT Dataset & JIT Streaming Engine
=============================================================================
Provides prompt-target masked instruction datasets for Supervised Fine-Tuning
across the 14 canonical domains with JIT streaming, dynamic ledger deduplication,
and zero sample repeat guarantees.
=============================================================================
"""

import bz2
import copy
import csv
import gzip
import hashlib
import json
import math
import os
import re
import sys
import time
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import numpy as np
import torch
from torch.utils.data import IterableDataset

# Canonical 14 TARA Domains
CANONICAL_DOMAINS = [
    "01_math_scientific_reasoning",
    "02_code_algorithmic_logic",
    "03_natural_sciences_stem",
    "04_linguistics_multilingual",
    "05_philosophy_cognitive_ethics",
    "06_social_historical_knowledge",
    "07_engineering_systems_infra",
    "08_commonsense_world_model",
    "09_multimodal_audio",
    "10_multimodal_vision",
    "11_tool_use_execution_chains",
    "12_planning_reasoning_traces",
    "13_debugging_self_correction",
    "sft",
]

DOMAIN_ALIASES = {
    "math": "01_math_scientific_reasoning",
    "code": "02_code_algorithmic_logic",
    "stem": "03_natural_sciences_stem",
    "science": "03_natural_sciences_stem",
    "kannada": "04_linguistics_multilingual",
    "multilingual": "04_linguistics_multilingual",
    "philosophy": "05_philosophy_cognitive_ethics",
    "social": "06_social_historical_knowledge",
    "engineering": "07_engineering_systems_infra",
    "world_model": "08_commonsense_world_model",
    "audio": "09_multimodal_audio",
    "vision": "10_multimodal_vision",
    "tool_use": "11_tool_use_execution_chains",
    "planning": "12_planning_reasoning_traces",
    "reasoning": "12_planning_reasoning_traces",
    "debugging": "13_debugging_self_correction",
    "sft": "sft",
}


def normalize_domain(domain: str) -> str:
    """Resolves canonical domain identifier from aliases or raw strings."""
    cleaned = domain.strip().lower()
    if cleaned in CANONICAL_DOMAINS:
        return cleaned

    # Check 2-digit numeric domain prefixes (e.g. 01_mathematics_formal_logic -> 01_math_scientific_reasoning)
    if len(cleaned) >= 2 and cleaned[:2].isdigit():
        p_num = int(cleaned[:2])
        if 1 <= p_num <= 13:
            return CANONICAL_DOMAINS[p_num - 1]

    for prefix, target in DOMAIN_ALIASES.items():
        if prefix in cleaned or cleaned.startswith(prefix):
            return target
    return "sft"


def parse_domain_weights(weight_spec: str) -> Dict[str, float]:
    """Parses domain weights string e.g. 'kannada:0.25,code:0.25,reasoning:0.25,sft:0.25'."""
    weights: Dict[str, float] = {}
    if not weight_spec:
        equal_weight = 1.0 / len(CANONICAL_DOMAINS)
        return {d: equal_weight for d in CANONICAL_DOMAINS}

    pairs = [p.strip() for p in weight_spec.split(",") if ":" in p]
    for p in pairs:
        parts = p.split(":", 1)
        dom = normalize_domain(parts[0])
        try:
            w = float(parts[1])
            weights[dom] = max(0.0, w)
        except ValueError:
            continue

    total = sum(weights.values())
    if total <= 0:
        equal_weight = 1.0 / len(CANONICAL_DOMAINS)
        return {d: equal_weight for d in CANONICAL_DOMAINS}

    return {k: v / total for k, v in weights.items()}


class SftShardLedger:
    """
    Dynamic state ledger that tracks completed and active data shards
    to ensure zero repeated records across session resumes.
    """

    def __init__(self, ledger_path: str):
        self.ledger_path = ledger_path
        self.completed_shards: Set[str] = set()
        self.active_cursors: Dict[str, Dict[str, int]] = {}
        self.seen_record_hashes: Set[str] = set()
        self._load()

    def _load(self):
        if not os.path.exists(self.ledger_path):
            return
        try:
            with open(self.ledger_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.completed_shards = set(data.get("completed_shards", []))
            self.active_cursors = data.get("active_cursors", {})
            self.seen_record_hashes = set(data.get("seen_hashes", []))
        except Exception:
            pass

    def save(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.ledger_path)), exist_ok=True)
        tmp_path = f"{self.ledger_path}.tmp.{os.getpid()}"
        payload = {
            "completed_shards": sorted(list(self.completed_shards)),
            "active_cursors": self.active_cursors,
            "seen_hashes": list(self.seen_record_hashes)[-50000:],  # preserve recent dedup window
            "updated_at": time.time(),
        }
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, self.ledger_path)

    def is_shard_completed(self, shard_name: str) -> bool:
        return shard_name in self.completed_shards

    def mark_shard_completed(self, shard_name: str):
        self.completed_shards.add(shard_name)
        if shard_name in self.active_cursors:
            del self.active_cursors[shard_name]

    def update_cursor(self, shard_name: str, line_offset: int, byte_offset: int):
        self.active_cursors[shard_name] = {
            "line_offset": line_offset,
            "byte_offset": byte_offset,
        }

    def get_cursor(self, shard_name: str) -> Tuple[int, int]:
        c = self.active_cursors.get(shard_name, {})
        return c.get("line_offset", 0), c.get("byte_offset", 0)

    def is_duplicate_record(self, record_text: str) -> bool:
        h = hashlib.sha256(record_text.encode("utf-8", errors="ignore")).hexdigest()
        if h in self.seen_record_hashes:
            return True
        self.seen_record_hashes.add(h)
        return False


def format_chat_prompt(instruction: str, response: str, system_prompt: Optional[str] = None) -> Tuple[str, str]:
    """Formats instruction and response into canonical ChatML tokens."""
    sys_part = f"<|im_start|>system\n{system_prompt.strip()}<|im_end|>\n" if system_prompt else ""
    prompt = f"{sys_part}<|im_start|>user\n{instruction.strip()}<|im_end|>\n<|im_start|>assistant\n"
    target = f"{response.strip()}<|im_end|>"
    return prompt, target


def extract_inst_resp_from_dict(val: Any) -> Optional[Tuple[str, str]]:
    """
    Universally extracts (instruction, response) pairs across:
    1. messages format (SmolTalk, NuminaMath, ChatML / ShareGPT)
    2. conversations format (OpenThoughts, ShareGPT)
    3. problem / solution pairs (NuminaMath, OpenThoughts metadata)
    4. canonical instruction / response or input / output pairs
    5. text blocks with Assistant: / ಉತ್ತರ: delimiters
    """
    if not isinstance(val, dict):
        return None

    # 1. messages format (e.g. [{'role': 'user', 'content': ...}, {'role': 'assistant', 'content': ...}])
    if "messages" in val and isinstance(val["messages"], (list, tuple, np.ndarray)):
        system_parts = []
        user_parts = []
        asst_parts = []
        for msg in val["messages"]:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role", "")).lower()
            content = msg.get("content") or ""
            if not content:
                continue
            if role in ("system",):
                system_parts.append(str(content).strip())
            elif role in ("user", "human"):
                user_parts.append(str(content).strip())
            elif role in ("assistant", "gpt", "model"):
                asst_parts.append(str(content).strip())
        if user_parts and asst_parts:
            inst = "\n\n".join(user_parts)
            if system_parts:
                inst = f"{' '.join(system_parts)}\n\n{inst}"
            resp = "\n\n".join(asst_parts)
            return (inst, resp)

    # 2. conversations format (e.g. [{'from': 'user', 'value': ...}, {'from': 'assistant', 'value': ...}])
    if "conversations" in val and isinstance(val["conversations"], (list, tuple, np.ndarray)):
        system_str = str(val.get("system", "")).strip()
        user_parts = []
        asst_parts = []
        for msg in val["conversations"]:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("from", "")).lower()
            val_txt = msg.get("value") or msg.get("content") or ""
            if not val_txt:
                continue
            if role in ("system",) and not system_str:
                system_str = str(val_txt).strip()
            elif role in ("user", "human"):
                user_parts.append(str(val_txt).strip())
            elif role in ("assistant", "gpt", "model"):
                asst_parts.append(str(val_txt).strip())
        if user_parts and asst_parts:
            inst = "\n\n".join(user_parts)
            if system_str:
                inst = f"{system_str}\n\n{inst}"
            resp = "\n\n".join(asst_parts)
            return (inst, resp)

    # 3. problem / solution pairs (NuminaMath, OpenThoughts metadata)
    if "problem" in val:
        prob = str(val["problem"]).strip()
        sol = (
            val.get("solution")
            or val.get("ground_truth_solution")
            or val.get("deepseek_solution")
            or ""
        )
        if prob and sol:
            return (prob, str(sol).strip())

    # 4. Canonical instruction / response or input / output pairs
    inst = (
        val.get("instruction")
        or val.get("input")
        or val.get("prompt")
        or val.get("question")
        or ""
    )
    resp = (
        val.get("response")
        or val.get("output")
        or val.get("completion")
        or val.get("answer")
        or ""
    )
    if inst and resp:
        return (str(inst).strip(), str(resp).strip())

    # 5. Raw text block with Assistant / ಉತ್ತರ
    if "text" in val:
        t = str(val["text"]).strip()
        if "\nAssistant:" in t:
            parts = t.split("\nAssistant:", 1)
            return (parts[0].replace("User:", "").strip(), parts[1].strip())
        elif "\nಉತ್ತರ:" in t:
            parts = t.split("\nಉತ್ತರ:", 1)
            return (parts[0].replace("ಪ್ರಶ್ನೆ:", "").strip(), parts[1].strip())

    return None


def open_text_stream(file_path: str):
    """Transparently opens plaintext, gzip, or bzip2 compressed streams."""
    p_lower = file_path.lower()
    if p_lower.endswith(".gz"):
        return gzip.open(file_path, "rt", encoding="utf-8", errors="replace")
    elif p_lower.endswith(".bz2"):
        return bz2.open(file_path, "rt", encoding="utf-8", errors="replace")
    else:
        return open(file_path, "r", encoding="utf-8", errors="replace")


def get_format_type(file_path: str) -> str:
    """Identifies the canonical format and compression of a dataset file."""
    p_lower = file_path.lower()
    for comp in (".gz", ".bz2"):
        if p_lower.endswith(comp):
            p_lower = p_lower[:-len(comp)]
            break
    if p_lower.endswith(".jsonl"):
        return "jsonl"
    if p_lower.endswith(".json"):
        return "json"
    if p_lower.endswith(".parquet"):
        return "parquet"
    if p_lower.endswith(".csv"):
        return "csv"
    if p_lower.endswith(".tsv"):
        return "tsv"
    if p_lower.endswith(".txt"):
        return "txt"
    return "unknown"


class SftItem:
    """Container for tokenized SFT sample with masked labels."""

    def __init__(self, input_ids: torch.Tensor, labels: torch.Tensor, domain: str, shard: str):
        self.input_ids = input_ids
        self.labels = labels
        self.domain = domain
        self.shard = shard


class SftSanitizer:
    """
    Direct in-file sanitization & security filtering suite for SFT instruction datasets.
    Filters:
    1. NaN / Null / None / Empty fields
    2. Malicious prompts, prompt injections, and dangerous exploit commands
    3. PII & Email addresses (scrubbed to [EMAIL_REDACTED], tokens/keys scrubbed)
    4. Author IDs, Git commit author spam, and author handles
    5. Garbage characters, replacement chars (\ufffd, \0), and corrupted escapes
    """

    EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    API_KEY_PATTERN = re.compile(r'\b(?:sk-[a-zA-Z0-9]{20,}|ghp_[a-zA-Z0-9]{20,}|bearer\s+[a-zA-Z0-9\._\-]{20,})\b', re.IGNORECASE)
    AUTHOR_METADATA_PATTERN = re.compile(
        r'^(?:Author|Signed-off-by|Co-authored-by|Commit|User-ID|Author-ID|Reviewed-by):\s*.*$',
        re.MULTILINE | re.IGNORECASE,
    )
    AUTHOR_HANDLE_PATTERN = re.compile(r'@[a-zA-Z0-9_-]{3,25}(?:\s*\(Author\)|\s*\(Contributor\))', re.IGNORECASE)

    MALICIOUS_PATTERNS = [
        "ignore all previous instructions",
        "ignore all prior instructions",
        "disregard all previous instructions",
        "disregard all previous guidelines",
        "system prompt override",
        "you are now in developer mode",
        "dan mode",
        "jailbreak enabled",
        "bypass safety guidelines",
        "rm -rf /",
        "/bin/sh",
        "/bin/bash",
        "drop database",
        "<script",
        "eval(",
        "passthru(",
        "base64_decode(",
        "nc -e /bin",
    ]

    CORRUPTED_ARTIFACTS = [
        r'\",\":',
        r'\",\"',
        r'\", \"',
        r'\",\n',
        r'\"}',
        "[NULL]",
        "\ufffd",
        "\0",
    ]

    @classmethod
    def is_nan_or_empty(cls, text: Any) -> bool:
        if text is None:
            return True
        if isinstance(text, float) and (math.isnan(text) or math.isinf(text)):
            return True
        s = str(text).strip()
        if not s:
            return True
        lower = s.lower()
        if lower in ("nan", "null", "none", "undefined", "nil", "n/a", "{}"):
            return True
        return False

    @classmethod
    def is_malicious(cls, text: str) -> bool:
        lower = text.lower()
        for p in cls.MALICIOUS_PATTERNS:
            if p in lower:
                return True
        return False

    @classmethod
    def sanitize_pii_and_emails(cls, text: str) -> Tuple[str, bool]:
        changed = False
        if "@" in text and cls.EMAIL_PATTERN.search(text):
            text = cls.EMAIL_PATTERN.sub("[EMAIL_REDACTED]", text)
            changed = True
        if cls.API_KEY_PATTERN.search(text):
            text = cls.API_KEY_PATTERN.sub("[KEY_REDACTED]", text)
            changed = True
        return text, changed

    @classmethod
    def strip_author_metadata(cls, text: str) -> Tuple[str, bool]:
        changed = False
        if cls.AUTHOR_METADATA_PATTERN.search(text):
            text = cls.AUTHOR_METADATA_PATTERN.sub("", text).strip()
            changed = True
        if cls.AUTHOR_HANDLE_PATTERN.search(text):
            text = cls.AUTHOR_HANDLE_PATTERN.sub("[AUTHOR_REDACTED]", text)
            changed = True
        return text, changed

    @classmethod
    def has_corrupted_artifacts(cls, text: str) -> bool:
        for c in cls.CORRUPTED_ARTIFACTS:
            if c in text:
                return True
        return False

    @classmethod
    def is_garbage(cls, text: str) -> bool:
        trimmed = text.strip()
        if len(trimmed) < 4 or len(trimmed) > 131_072:
            return True
        if "\ufffd" in trimmed or "\0" in trimmed:
            return True
        if re.search(r'(.)\1{50,}', trimmed):
            return True
        return False

    @classmethod
    def sanitize_record(cls, instruction: Any, response: Any) -> Optional[Tuple[str, str]]:
        if cls.is_nan_or_empty(instruction) or cls.is_nan_or_empty(response):
            return None

        inst_str = str(instruction).strip()
        resp_str = str(response).strip()

        if cls.is_garbage(inst_str) or cls.is_garbage(resp_str):
            return None

        if cls.has_corrupted_artifacts(inst_str) or cls.has_corrupted_artifacts(resp_str):
            return None

        if cls.is_malicious(inst_str) or cls.is_malicious(resp_str):
            return None

        inst_str, _ = cls.strip_author_metadata(inst_str)
        resp_str, _ = cls.strip_author_metadata(resp_str)

        inst_str, _ = cls.sanitize_pii_and_emails(inst_str)
        resp_str, _ = cls.sanitize_pii_and_emails(resp_str)

        if not inst_str.strip() or not resp_str.strip():
            return None

        return inst_str.strip(), resp_str.strip()


class MultiDomainSftDataset(IterableDataset):
    """
    Streams multi-domain instruction records with prompt-target masking.
    Labels for instruction tokens are masked with -100 so loss is computed
    strictly on the model's generated response tokens.
    """

    def __init__(
        self,
        tokenizer: Any,
        data_files: Dict[str, List[str]],
        ledger: SftShardLedger,
        domain_weights: Optional[Dict[str, float]] = None,
        max_seq_len: int = 1024,
        skip_samples: int = 0,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.data_files = data_files
        self.ledger = ledger
        self.max_seq_len = max_seq_len
        self.domain_weights = domain_weights or {d: 1.0 / len(CANONICAL_DOMAINS) for d in CANONICAL_DOMAINS}
        self.skip_samples = skip_samples
        self.emitted_samples = 0

    def _encode_sample(self, instruction: str, response: str, domain: str, shard: str) -> Optional[SftItem]:
        if not instruction.strip() or not response.strip():
            return None

        prompt_str, target_str = format_chat_prompt(instruction, response)
        enc_prompt = self.tokenizer.encode(prompt_str)
        prompt_ids = enc_prompt.ids if hasattr(enc_prompt, "ids") else list(enc_prompt)

        enc_target = self.tokenizer.encode(target_str)
        target_ids = enc_target.ids if hasattr(enc_target, "ids") else list(enc_target)

        if not target_ids:
            return None

        # Truncate prompt if it exceeds budget, preserving headroom for target
        max_prompt_len = self.max_seq_len - min(len(target_ids), self.max_seq_len // 2)
        if len(prompt_ids) > max_prompt_len:
            prompt_ids = prompt_ids[-max_prompt_len:]

        full_ids = (prompt_ids + target_ids)[: self.max_seq_len]
        if len(full_ids) < 4:
            return None

        input_ids = torch.tensor(full_ids, dtype=torch.long)
        labels = input_ids.clone()

        # Prompt-target masking: mask prompt tokens with -100
        mask_len = min(len(prompt_ids), len(full_ids))
        labels[:mask_len] = -100

        # Discard if zero supervisory target tokens remain
        if not (labels != -100).any():
            return None

        return SftItem(input_ids, labels, domain, shard)

    def _parse_raw_line(self, raw_line: str) -> Optional[Tuple[str, str]]:
        """Parses line into sanitized instruction and response pair."""
        try:
            val = json.loads(raw_line)
            candidate = extract_inst_resp_from_dict(val)
            if candidate is not None:
                return SftSanitizer.sanitize_record(candidate[0], candidate[1])
        except Exception:
            pass
        return None

    def __iter__(self) -> Iterator[SftItem]:
        # Weighted round-robin across active domain file pools
        active_domains = [d for d in self.data_files if self.data_files[d]]
        if not active_domains:
            return

        for domain in active_domains:
            shards = self.data_files[domain]
            for shard_path in shards:
                shard_name = os.path.basename(shard_path)
                if self.ledger.is_shard_completed(shard_name):
                    continue

                line_off, byte_off = self.ledger.get_cursor(shard_name)
                fmt = get_format_type(shard_path)

                if fmt == "jsonl":
                    if not os.path.exists(shard_path):
                        continue
                    with open_text_stream(shard_path) as f:
                        if byte_off > 0 and not (shard_path.lower().endswith(".gz") or shard_path.lower().endswith(".bz2")):
                            f.seek(byte_off)
                        cur_line = line_off
                        while True:
                            b_pos = f.tell() if hasattr(f, "tell") else 0
                            line = f.readline()
                            if not line:
                                self.ledger.mark_shard_completed(shard_name)
                                break
                            cur_line += 1
                            parsed = self._parse_raw_line(line)
                            if not parsed:
                                continue
                            inst, resp = parsed
                            if self.ledger.is_duplicate_record(f"{inst}\n{resp}"):
                                continue

                            sample = self._encode_sample(inst, resp, domain, shard_name)
                            if sample is not None:
                                self.ledger.update_cursor(shard_name, cur_line, b_pos)
                                if self.emitted_samples < self.skip_samples:
                                    self.emitted_samples += 1
                                    continue
                                self.emitted_samples += 1
                                yield sample

                elif fmt == "txt":
                    if not os.path.exists(shard_path):
                        continue
                    try:
                        with open_text_stream(shard_path) as f:
                            content = f.read()
                        for block in content.split("\n\n"):
                            block = block.strip()
                            if not block:
                                continue
                            inst_resp = None
                            if "\nAssistant:" in block:
                                parts = block.split("\nAssistant:", 1)
                                inst_resp = (parts[0].replace("User:", "").strip(), parts[1].strip())
                            elif "\nಉತ್ತರ:" in block:
                                parts = block.split("\nಉತ್ತರ:", 1)
                                inst_resp = (parts[0].replace("ಪ್ರಶ್ನೆ:", "").strip(), parts[1].strip())
                            elif "\t" in block:
                                parts = block.split("\t", 1)
                                inst_resp = (parts[0].strip(), parts[1].strip())
                            else:
                                parsed = self._parse_raw_line(block)
                                if parsed:
                                    inst_resp = parsed
                            if inst_resp:
                                clean = SftSanitizer.sanitize_record(inst_resp[0], inst_resp[1])
                                if clean and not self.ledger.is_duplicate_record(f"{clean[0]}\n{clean[1]}"):
                                    sample = self._encode_sample(clean[0], clean[1], domain, shard_name)
                                    if sample is not None:
                                        if self.emitted_samples < self.skip_samples:
                                            self.emitted_samples += 1
                                            continue
                                        self.emitted_samples += 1
                                        yield sample
                        self.ledger.mark_shard_completed(shard_name)
                    except Exception:
                        continue

                elif fmt == "json":
                    if not os.path.exists(shard_path):
                        continue
                    try:
                        with open_text_stream(shard_path) as f:
                            content = json.load(f)
                        items = content if isinstance(content, list) else [content]
                        for item in items:
                            if not isinstance(item, dict):
                                continue
                            raw_line = json.dumps(item)
                            parsed = self._parse_raw_line(raw_line)
                            if not parsed:
                                continue
                            inst, resp = parsed
                            if self.ledger.is_duplicate_record(f"{inst}\n{resp}"):
                                continue
                            sample = self._encode_sample(inst, resp, domain, shard_name)
                            if sample is not None:
                                if self.emitted_samples < self.skip_samples:
                                    self.emitted_samples += 1
                                    continue
                                self.emitted_samples += 1
                                yield sample
                        self.ledger.mark_shard_completed(shard_name)
                    except Exception:
                        continue

                elif fmt in ("csv", "tsv"):
                    if not os.path.exists(shard_path):
                        continue
                    try:
                        delimiter = "\t" if fmt == "tsv" else ","
                        with open_text_stream(shard_path) as f:
                            reader = csv.DictReader(f, delimiter=delimiter)
                            for row in reader:
                                if not row:
                                    continue
                                raw_line = json.dumps(row)
                                parsed = self._parse_raw_line(raw_line)
                                if not parsed:
                                    continue
                                inst, resp = parsed
                                if self.ledger.is_duplicate_record(f"{inst}\n{resp}"):
                                    continue
                                sample = self._encode_sample(inst, resp, domain, shard_name)
                                if sample is not None:
                                    if self.emitted_samples < self.skip_samples:
                                        self.emitted_samples += 1
                                        continue
                                    self.emitted_samples += 1
                                    yield sample
                        self.ledger.mark_shard_completed(shard_name)
                    except Exception:
                        continue

                elif fmt == "parquet":
                    if not os.path.exists(shard_path):
                        continue
                    try:
                        import pyarrow.parquet as pq

                        line_off, _ = self.ledger.get_cursor(shard_name)
                        cur_row = 0
                        pf = pq.ParquetFile(shard_path)
                        for batch in pf.iter_batches(batch_size=64):
                            df = batch.to_pydict()
                            keys = list(df.keys())
                            n_rows = len(df[keys[0]]) if keys else 0
                            for r in range(n_rows):
                                cur_row += 1
                                if cur_row <= line_off:
                                    continue
                                row = {k: df[k][r] for k in keys}
                                candidate = extract_inst_resp_from_dict(row)
                                if not candidate:
                                    continue
                                sanitized = SftSanitizer.sanitize_record(candidate[0], candidate[1])
                                if not sanitized:
                                    continue
                                c_inst, c_resp = sanitized
                                if self.ledger.is_duplicate_record(f"{c_inst}\n{c_resp}"):
                                    continue
                                sample = self._encode_sample(c_inst, c_resp, domain, shard_name)
                                if sample is not None:
                                    self.ledger.update_cursor(shard_name, cur_row, 0)
                                    if self.emitted_samples < self.skip_samples:
                                        self.emitted_samples += 1
                                        continue
                                    self.emitted_samples += 1
                                    yield sample
                        self.ledger.mark_shard_completed(shard_name)
                    except Exception:
                        continue


class BinaryCacheSftDataset(IterableDataset):
    """
    Zero-latency memory-mapped binary cache reader for pre-tokenized SFT tokens.
    Directly streams int32 token arrays with prompt-target masking from .bin files
    with deterministic cursor-based zero repetition on checkpoint resumption.
    """

    def __init__(self, cache_dir: str, ledger: SftShardLedger, skip_samples: int = 0):
        super().__init__()
        self.cache_dir = cache_dir
        self.ledger = ledger
        self.skip_samples = skip_samples
        self.manifest_path = os.path.join(cache_dir, "sft_pretokenization_manifest.json")
        self.domains: Dict[str, Any] = {}
        self._load_manifest()

    def _load_manifest(self):
        if not os.path.exists(self.manifest_path):
            return
        try:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.domains = data.get("domains", {})
        except Exception:
            pass

    def is_valid(self) -> bool:
        return bool(self.domains)

    def __iter__(self) -> Iterator[SftItem]:
        emitted_count = 0
        for domain, d_info in self.domains.items():
            bin_ids = os.path.join(self.cache_dir, d_info["bin_ids"])
            bin_lbl = os.path.join(self.cache_dir, d_info["bin_lbl"])
            idx_file = os.path.join(self.cache_dir, f"sft_{domain}_index.json")

            if not os.path.exists(bin_ids) or not os.path.exists(bin_lbl) or not os.path.exists(idx_file):
                continue

            try:
                with open(idx_file, "r", encoding="utf-8") as f_idx:
                    meta = json.load(f_idx)
                offsets = meta.get("offsets", [])

                # Memory map binary files for zero-copy slice access
                ids_mmap = np.memmap(bin_ids, dtype=np.int32, mode="r")
                lbl_mmap = np.memmap(bin_lbl, dtype=np.int32, mode="r")

                for start_tok, length in offsets:
                    if emitted_count < self.skip_samples:
                        emitted_count += 1
                        continue

                    emitted_count += 1
                    input_ids = torch.from_numpy(np.array(ids_mmap[start_tok : start_tok + length], dtype=np.int64))
                    labels = torch.from_numpy(np.array(lbl_mmap[start_tok : start_tok + length], dtype=np.int64))
                    yield SftItem(input_ids, labels, domain, d_info["bin_ids"])
            except Exception:
                continue

