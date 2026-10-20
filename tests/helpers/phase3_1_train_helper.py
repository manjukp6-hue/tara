#!/usr/bin/env python3
"""
Phase 3.1 subprocess helper for Gate C2: mid-shard interrupt + fresh-process resume.

Usage:
  python phase3_1_train_helper.py process_a --shard PATH --workdir DIR --stop-at-line N
  python phase3_1_train_helper.py process_b --shard PATH --workdir DIR

Process A:
  Streams the shard until line N is yielded, records the cursor to cursor_a.json,
  and writes units_a.json (list of [line_idx, chunk_idx] for units streamed).

Process B:
  Loads cursor_a.json, resumes the dataset from that cursor, and writes
  units_b.json (list of [line_idx, chunk_idx] for units yielded after resume).

No GPU training, no model, no optimizer — pure dataset iterator testing.
"""

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cloud_train import TaraNativeTokenizer, TaraStreamingJsonlDataset, compute_file_sha256


def _make_tokenizer():
    vocab = {chr(i): i for i in range(32, 64)}
    vocab["<|unk|>"] = 0
    return TaraNativeTokenizer(vocab)


def process_a(shard_path: str, workdir: str, stop_at_line: int):
    """
    Stream the shard. Record all units up to and including the last unit
    whose line_idx < stop_at_line. Write cursor and unit list to workdir.
    """
    tokenizer = _make_tokenizer()
    shard_name = os.path.basename(shard_path)
    dataset_dir = os.path.dirname(shard_path)

    ds = TaraStreamingJsonlDataset(
        file_paths=[shard_name],
        tokenizer=tokenizer,
        max_seq_len=8,
        dataset_dir=dataset_dir,
    )

    units_a = []
    last_committed_line = -1
    last_committed_chunk = 0

    for item in ds:
        # item: (input_ids, labels, shard_name, line_idx, byte_offset, chunk_idx)
        line_idx = item[3]
        chunk_idx = item[5]
        byte_off = item[4]

        if line_idx >= stop_at_line:
            # Stop — do not commit this unit
            break

        units_a.append([line_idx, chunk_idx])
        last_committed_line = line_idx
        last_committed_chunk = chunk_idx
        last_byte_offset = byte_off

    # cursor_a represents: last committed position
    # resume_chunk_idx = last_committed_chunk + 1 (next chunk to train in that line)
    # If last_committed_chunk was the last chunk of that line, the dataset will advance
    # to the next line automatically.
    cursor_a = {
        "resume_shard": shard_name,
        "resume_line_offset": last_committed_line,
        "resume_chunk_idx": last_committed_chunk + 1,
        "resume_byte_offset": last_byte_offset if last_committed_line >= 0 else 0,
    }

    with open(os.path.join(workdir, "cursor_a.json"), "w", encoding="utf-8") as f:
        json.dump(cursor_a, f, indent=2)
    with open(os.path.join(workdir, "units_a.json"), "w", encoding="utf-8") as f:
        json.dump(units_a, f)

    print(f"[Process A] Committed {len(units_a)} units. Cursor: line={cursor_a['resume_line_offset']}, chunk={cursor_a['resume_chunk_idx']}")


def process_b(shard_path: str, workdir: str):
    """
    Load cursor from Process A. Resume dataset. Write all yielded units to units_b.json.
    """
    cursor_file = os.path.join(workdir, "cursor_a.json")
    if not os.path.exists(cursor_file):
        print(f"[Process B] cursor_a.json not found at {cursor_file}", file=sys.stderr)
        sys.exit(1)

    with open(cursor_file, encoding="utf-8") as f:
        cursor = json.load(f)

    tokenizer = _make_tokenizer()
    shard_name = os.path.basename(shard_path)
    dataset_dir = os.path.dirname(shard_path)

    ds = TaraStreamingJsonlDataset(
        file_paths=[shard_name],
        tokenizer=tokenizer,
        max_seq_len=8,
        dataset_dir=dataset_dir,
        resume_shard=cursor["resume_shard"],
        resume_line_offset=cursor["resume_line_offset"],
        resume_chunk_idx=cursor["resume_chunk_idx"],
        resume_byte_offset=cursor["resume_byte_offset"],
    )

    units_b = [[item[3], item[5]] for item in ds]  # [line_idx, chunk_idx]

    with open(os.path.join(workdir, "units_b.json"), "w", encoding="utf-8") as f:
        json.dump(units_b, f)

    print(f"[Process B] Resumed and yielded {len(units_b)} units. First: {units_b[0] if units_b else 'NONE'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["process_a", "process_b"])
    parser.add_argument("--shard", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--stop-at-line", type=int, default=3)
    args = parser.parse_args()

    if args.mode == "process_a":
        process_a(args.shard, args.workdir, args.stop_at_line)
    else:
        process_b(args.shard, args.workdir)
