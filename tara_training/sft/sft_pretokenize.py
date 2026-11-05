#!/usr/bin/env python3
"""
TARA SFT Pre-Tokenization & Binary Cache Engine
=============================================================================
Recursively scans, filters, and pre-tokenizes multi-domain instruction datasets
(JSONL, Parquet, Text) into zero-latency memory-mapped binary cache files (.bin)
prior to training, eliminating CPU bottlenecks during GPU optimization.
=============================================================================
"""

import argparse
import bz2
import csv
import gzip
import hashlib
import io
import json
import os
import struct
import sys
import tarfile
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import zipfile

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(TRAINING_DIR)

for p in (REPO_ROOT, TRAINING_DIR, SCRIPT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from model_dataset import TaraNativeTokenizer
from sft_dataset import (
    CANONICAL_DOMAINS,
    SftSanitizer,
    extract_inst_resp_from_dict,
    format_chat_prompt,
    normalize_domain,
)


VALID_EXTENSIONS = (
    ".jsonl", ".json", ".parquet", ".csv", ".tsv", ".txt",
    ".jsonl.gz", ".json.gz", ".csv.gz", ".tsv.gz", ".txt.gz",
    ".jsonl.bz2", ".json.bz2", ".csv.bz2", ".tsv.bz2", ".txt.bz2",
    ".zip", ".tar.gz", ".tgz",
)


def get_format_type(file_path: str) -> str:
    """Identifies the canonical format and compression of a dataset file."""
    p_lower = file_path.lower()
    if p_lower.endswith((".tar.gz", ".tgz")):
        return "tar"
    if p_lower.endswith(".zip"):
        return "zip"
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


def open_text_stream(file_path: str):
    """Transparently opens plaintext, gzip, or bzip2 compressed streams."""
    p_lower = file_path.lower()
    if p_lower.endswith(".gz"):
        return gzip.open(file_path, "rt", encoding="utf-8", errors="replace")
    elif p_lower.endswith(".bz2"):
        return bz2.open(file_path, "rt", encoding="utf-8", errors="replace")
    else:
        return open(file_path, "r", encoding="utf-8", errors="replace")


def scan_all_subfolders_recursive(data_dir: str) -> Dict[str, List[str]]:
    """Deep recursive scan across all nested subdirectories for all 14 canonical domains."""
    domain_files: Dict[str, List[str]] = {d: [] for d in CANONICAL_DOMAINS}

    if not os.path.exists(data_dir):
        return domain_files

    for root, _, files in os.walk(data_dir):
        for f in files:
            f_lower = f.lower()
            if any(f_lower.endswith(ext) for ext in VALID_EXTENSIONS):
                full_path = os.path.join(root, f)
                rel_parts = os.path.relpath(full_path, data_dir).split(os.sep)
                assigned = "sft"
                for p in rel_parts:
                    norm = normalize_domain(p)
                    if norm != "sft":
                        assigned = norm
                        break
                domain_files[assigned].append(full_path)

    for d in domain_files:
        domain_files[d].sort()

    return domain_files


def extract_instruction_response_pairs(file_path: str) -> List[Tuple[str, str]]:
    """
    Extracts raw instruction-response pairs across diverse formats & compressions:
    JSONL, JSON, Parquet, CSV, TSV, TXT, .gz, .bz2, .zip, .tar.gz / .tgz.
    """
    pairs: List[Tuple[str, str]] = []
    fmt = get_format_type(file_path)

    if fmt == "jsonl":
        try:
            with open_text_stream(file_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        val = json.loads(line)
                        pair = extract_inst_resp_from_dict(val)
                        if pair:
                            pairs.append(pair)
                    except Exception:
                        continue
        except Exception:
            pass

    elif fmt == "txt":
        try:
            with open_text_stream(file_path) as f:
                content = f.read()
            for block in content.split("\n\n"):
                block = block.strip()
                if not block:
                    continue
                if "\nAssistant:" in block:
                    parts = block.split("\nAssistant:", 1)
                    pairs.append((parts[0].replace("User:", "").strip(), parts[1].strip()))
                elif "\nಉತ್ತರ:" in block:
                    parts = block.split("\nಉತ್ತರ:", 1)
                    pairs.append((parts[0].replace("ಪ್ರಶ್ನೆ:", "").strip(), parts[1].strip()))
                elif "\t" in block:
                    parts = block.split("\t", 1)
                    pairs.append((parts[0].strip(), parts[1].strip()))
                else:
                    try:
                        val = json.loads(block)
                        pair = extract_inst_resp_from_dict(val)
                        if pair:
                            pairs.append(pair)
                    except Exception:
                        pass
        except Exception:
            pass

    elif fmt == "json":
        try:
            with open_text_stream(file_path) as f:
                content = json.load(f)
            items = content if isinstance(content, list) else [content]
            for val in items:
                if not isinstance(val, dict):
                    continue
                pair = extract_inst_resp_from_dict(val)
                if pair:
                    pairs.append(pair)
        except Exception:
            pass

    elif fmt in ("csv", "tsv"):
        try:
            delimiter = "\t" if fmt == "tsv" else ","
            with open_text_stream(file_path) as f:
                reader = csv.DictReader(f, delimiter=delimiter)
                for row in reader:
                    if not row:
                        continue
                    pair = extract_inst_resp_from_dict(row)
                    if pair:
                        pairs.append(pair)
        except Exception:
            pass

    elif fmt == "parquet":
        try:
            import pyarrow.parquet as pq

            pf = pq.ParquetFile(file_path)
            for batch in pf.iter_batches(batch_size=256):
                df = batch.to_pydict()
                keys = list(df.keys())
                n_rows = len(df[keys[0]]) if keys else 0
                for r in range(n_rows):
                    row = {k: df[k][r] for k in keys}
                    pair = extract_inst_resp_from_dict(row)
                    if pair:
                        pairs.append(pair)
        except Exception:
            pass

    elif fmt == "zip":
        try:
            with zipfile.ZipFile(file_path, "r") as zf:
                for member in zf.namelist():
                    m_lower = member.lower()
                    if m_lower.endswith((".jsonl", ".txt")):
                        with zf.open(member) as mf:
                            for line_b in mf:
                                line = line_b.decode("utf-8", errors="replace").strip()
                                if not line:
                                    continue
                                try:
                                    val = json.loads(line)
                                    pair = extract_inst_resp_from_dict(val)
                                    if pair:
                                        pairs.append(pair)
                                except Exception:
                                    continue
                    elif m_lower.endswith(".json"):
                        with zf.open(member) as mf:
                            try:
                                content = json.loads(mf.read().decode("utf-8", errors="replace"))
                                items = content if isinstance(content, list) else [content]
                                for val in items:
                                    if isinstance(val, dict):
                                        pair = extract_inst_resp_from_dict(val)
                                        if pair:
                                            pairs.append(pair)
                            except Exception:
                                pass
                    elif m_lower.endswith((".csv", ".tsv")):
                        with zf.open(member) as mf:
                            delim = "\t" if m_lower.endswith(".tsv") else ","
                            reader = csv.DictReader(io.TextIOWrapper(mf, encoding="utf-8", errors="replace"), delimiter=delim)
                            for row in reader:
                                pair = extract_inst_resp_from_dict(row)
                                if pair:
                                    pairs.append(pair)
        except Exception:
            pass

    elif fmt == "tar":
        try:
            with tarfile.open(file_path, "r:*") as tf:
                for member in tf.getmembers():
                    if not member.isfile():
                        continue
                    m_lower = member.name.lower()
                    f_in = tf.extractfile(member)
                    if not f_in:
                        continue
                    if m_lower.endswith((".jsonl", ".txt")):
                        for line_b in f_in:
                            line = line_b.decode("utf-8", errors="replace").strip()
                            if not line:
                                continue
                            try:
                                val = json.loads(line)
                                pair = extract_inst_resp_from_dict(val)
                                if pair:
                                    pairs.append(pair)
                            except Exception:
                                continue
                    elif m_lower.endswith(".json"):
                        try:
                            content = json.loads(f_in.read().decode("utf-8", errors="replace"))
                            items = content if isinstance(content, list) else [content]
                            for val in items:
                                if isinstance(val, dict):
                                    pair = extract_inst_resp_from_dict(val)
                                    if pair:
                                        pairs.append(pair)
                        except Exception:
                            pass
                    elif m_lower.endswith((".csv", ".tsv")):
                        delim = "\t" if m_lower.endswith(".tsv") else ","
                        reader = csv.DictReader(io.TextIOWrapper(f_in, encoding="utf-8", errors="replace"), delimiter=delim)
                        for row in reader:
                            pair = extract_inst_resp_from_dict(row)
                            if pair:
                                pairs.append(pair)
        except Exception:
            pass

    return pairs


def pretokenize_domain(
    domain: str,
    files: List[str],
    tokenizer: Any,
    output_dir: str,
    max_seq_len: int = 1024,
    seen_hashes: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """Applies security, PII, author, and NaN filters before writing binary cache files."""
    if seen_hashes is None:
        seen_hashes = set()

    os.makedirs(output_dir, exist_ok=True)
    bin_ids_path = os.path.join(output_dir, f"sft_{domain}_input_ids.bin")
    bin_lbl_path = os.path.join(output_dir, f"sft_{domain}_labels.bin")
    idx_path = os.path.join(output_dir, f"sft_{domain}_index.json")

    sample_offsets: List[Tuple[int, int]] = []  # (token_offset, token_length)
    total_samples = 0
    total_tokens = 0

    # Filter telemetry tracking
    dropped_nan = 0
    dropped_garbage = 0
    dropped_corrupted = 0
    dropped_malicious = 0
    dropped_duplicate = 0
    emails_pii_scrubbed = 0
    author_ids_stripped = 0

    with open(bin_ids_path, "wb") as f_ids, open(bin_lbl_path, "wb") as f_lbl:
        for f_path in files:
            pairs = extract_instruction_response_pairs(f_path)
            for raw_inst, raw_resp in pairs:
                # 1. NaN, Null, Empty Check
                if SftSanitizer.is_nan_or_empty(raw_inst) or SftSanitizer.is_nan_or_empty(raw_resp):
                    dropped_nan += 1
                    continue

                # 2. Garbage & length bounds check
                if SftSanitizer.is_garbage(str(raw_inst)) or SftSanitizer.is_garbage(str(raw_resp)):
                    dropped_garbage += 1
                    continue

                # 3. Corrupted artifacts check
                if SftSanitizer.has_corrupted_artifacts(str(raw_inst)) or SftSanitizer.has_corrupted_artifacts(str(raw_resp)):
                    dropped_corrupted += 1
                    continue

                # 4. Malicious prompts & security exploits check
                if SftSanitizer.is_malicious(str(raw_inst)) or SftSanitizer.is_malicious(str(raw_resp)):
                    dropped_malicious += 1
                    continue

                # 5. Author IDs & Git commit author spam stripping
                inst, a_ch1 = SftSanitizer.strip_author_metadata(str(raw_inst))
                resp, a_ch2 = SftSanitizer.strip_author_metadata(str(raw_resp))
                if a_ch1 or a_ch2:
                    author_ids_stripped += 1

                # 6. PII & Email addresses scrubbing
                inst, p_ch1 = SftSanitizer.sanitize_pii_and_emails(inst)
                resp, p_ch2 = SftSanitizer.sanitize_pii_and_emails(resp)
                if p_ch1 or p_ch2:
                    emails_pii_scrubbed += 1

                if not inst.strip() or not resp.strip():
                    dropped_nan += 1
                    continue

                # Deduplication check
                content_hash = hashlib.sha256(f"{inst}\n{resp}".encode("utf-8")).hexdigest()
                if content_hash in seen_hashes:
                    dropped_duplicate += 1
                    continue
                seen_hashes.add(content_hash)

                prompt_str, target_str = format_chat_prompt(inst, resp)
                enc_p = tokenizer.encode(prompt_str)
                prompt_ids = enc_p.ids if hasattr(enc_p, "ids") else list(enc_p)

                enc_t = tokenizer.encode(target_str)
                target_ids = enc_t.ids if hasattr(enc_t, "ids") else list(enc_t)

                if not target_ids:
                    continue

                # Headroom truncation
                max_prompt_len = max_seq_len - min(len(target_ids), max_seq_len // 2)
                if len(prompt_ids) > max_prompt_len:
                    prompt_ids = prompt_ids[-max_prompt_len:]

                full_ids = (prompt_ids + target_ids)[:max_seq_len]
                if len(full_ids) < 4:
                    continue

                # Labels: prompt tokens masked to -100
                labels = list(full_ids)
                mask_len = min(len(prompt_ids), len(full_ids))
                for i in range(mask_len):
                    labels[i] = -100

                # Ensure supervisory target tokens exist
                if all(l == -100 for l in labels):
                    continue

                seq_len = len(full_ids)
                # Pack tokens as int32 (supports -100 for labels and up to 2B vocab)
                ids_bytes = struct.pack(f"<{seq_len}i", *full_ids)
                lbl_bytes = struct.pack(f"<{seq_len}i", *labels)

                f_ids.write(ids_bytes)
                f_lbl.write(lbl_bytes)

                sample_offsets.append((total_tokens, seq_len))
                total_tokens += seq_len
                total_samples += 1

    metadata = {
        "domain": domain,
        "sample_count": total_samples,
        "total_tokens": total_tokens,
        "input_ids_file": os.path.basename(bin_ids_path),
        "labels_file": os.path.basename(bin_lbl_path),
        "offsets": sample_offsets,
        "filter_telemetry": {
            "dropped_nan_empty": dropped_nan,
            "dropped_garbage": dropped_garbage,
            "dropped_corrupted": dropped_corrupted,
            "dropped_malicious": dropped_malicious,
            "dropped_duplicate": dropped_duplicate,
            "emails_pii_scrubbed": emails_pii_scrubbed,
            "author_ids_stripped": author_ids_stripped,
        },
        "created_at": time.time(),
    }

    with open(idx_path, "w", encoding="utf-8") as f_idx:
        json.dump(metadata, f_idx)

    return metadata


def main():
    parser = argparse.ArgumentParser(description="TARA SFT Pre-Tokenization & Binary Cache Compiler")
    parser.add_argument("--data-dir", type=str, required=True, help="Root dataset directory to scan recursively")
    parser.add_argument("--tokenizer-path", type=str, required=True, help="Path to tokenizer.json")
    parser.add_argument("--output-dir", type=str, default="storage/datasets/sft_binary_cache", help="Output directory for .bin files")
    parser.add_argument("--max-seq-len", type=int, default=1024, help="Maximum sequence context length")
    args = parser.parse_args()

    print("=" * 75, flush=True)
    print("  TARA SFT PRE-TOKENIZATION & BINARY CACHE COMPILER", flush=True)
    print("=" * 75, flush=True)
    print(f"  Source Dataset Root : {args.data_dir}", flush=True)
    print(f"  Tokenizer Path      : {args.tokenizer_path}", flush=True)
    print(f"  Binary Cache Output : {args.output_dir}", flush=True)
    print(f"  Context Max Length  : {args.max_seq_len}", flush=True)
    print("---------------------------------------------------------------------------", flush=True)

    if not os.path.exists(args.tokenizer_path):
        raise FileNotFoundError(f"Tokenizer missing: {args.tokenizer_path}")
    tokenizer = TaraNativeTokenizer.from_file(args.tokenizer_path)

    domain_files = scan_all_subfolders_recursive(args.data_dir)
    global_seen_hashes: Set[str] = set()
    manifest: Dict[str, Any] = {
        "manifest_version": "1.0",
        "created_at": time.time(),
        "domains": {},
        "total_samples": 0,
        "total_tokens": 0,
        "filter_telemetry": {
            "dropped_nan_empty": 0,
            "dropped_garbage": 0,
            "dropped_corrupted": 0,
            "dropped_malicious": 0,
            "dropped_duplicate": 0,
            "emails_pii_scrubbed": 0,
            "author_ids_stripped": 0,
        },
    }

    for domain in CANONICAL_DOMAINS:
        files = domain_files.get(domain, [])
        if not files:
            continue
        print(f"[Processing Domain] {domain} ({len(files)} files discovered)...", flush=True)
        meta = pretokenize_domain(
            domain=domain,
            files=files,
            tokenizer=tokenizer,
            output_dir=args.output_dir,
            max_seq_len=args.max_seq_len,
            seen_hashes=global_seen_hashes,
        )
        manifest["domains"][domain] = {
            "samples": meta["sample_count"],
            "tokens": meta["total_tokens"],
            "bin_ids": meta["input_ids_file"],
            "bin_lbl": meta["labels_file"],
            "filter_telemetry": meta["filter_telemetry"],
        }
        manifest["total_samples"] += meta["sample_count"]
        manifest["total_tokens"] += meta["total_tokens"]

        for k, v in meta["filter_telemetry"].items():
            manifest["filter_telemetry"][k] += v

        print(
            f"  -> Generated {meta['sample_count']:,} samples ({meta['total_tokens']:,} tokens) | "
            f"Malicious Dropped: {meta['filter_telemetry']['dropped_malicious']} | "
            f"PII Scrubbed: {meta['filter_telemetry']['emails_pii_scrubbed']} | "
            f"Author IDs: {meta['filter_telemetry']['author_ids_stripped']}",
            flush=True,
        )

    manifest_path = os.path.join(args.output_dir, "sft_pretokenization_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    t_tel = manifest["filter_telemetry"]
    print("=" * 75, flush=True)
    print(f"[Binary Cache Complete] Compiled {manifest['total_samples']:,} samples ({manifest['total_tokens']:,} tokens)", flush=True)
    print(
        f"[Filter Summary] Malicious: {t_tel['dropped_malicious']:,} | "
        f"NaN/Empty: {t_tel['dropped_nan_empty']:,} | "
        f"PII/Emails: {t_tel['emails_pii_scrubbed']:,} | "
        f"Author IDs: {t_tel['author_ids_stripped']:,} | "
        f"Duplicates: {t_tel['dropped_duplicate']:,}",
        flush=True,
    )
    print(f"Manifest written to: {manifest_path}", flush=True)
    print("=" * 75, flush=True)


if __name__ == "__main__":
    main()
