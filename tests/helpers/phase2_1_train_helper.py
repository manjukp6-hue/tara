#!/usr/bin/env python3
"""
Bounded synthetic training helper for Phase 2.1 integration tests.

Runs a self-contained training loop using the actual cloud_train functions
(TaraStreamingJsonlDataset, save_checkpoint, try_resume_from_checkpoint).
Records committed training unit IDs to a JSON log file so that cross-process
tests can assert on exactly which units were durably committed.

Safe token ID policy: only characters chr(32)..chr(63) are used in the
synthetic JSONL fixtures, mapping to embedding indices 32..63 which are all
within the tiny model's vocab_size=64 embedding table.

Exit codes
----------
0 — normal completion (all requested steps committed and checkpointed)
2 — intentional pre-checkpoint crash (simulated fault: optimizer step done,
    checkpoint NOT saved, process terminated)
3 — intentional post-checkpoint crash (checkpoint saved, latest-pointer NOT
    written, process terminated)

Usage
-----
python phase2_1_train_helper.py \\
    --output-dir /tmp/ckpt \\
    --shard-path /tmp/shard.jsonl \\
    --max-steps 3 \\
    --grad-accum 1 \\
    --log-file /tmp/log.json \\
    [--resume] \\
    [--crash-before-commit-step N] \\
    [--crash-after-commit-step N]
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

# Project root is two levels above this file (tests/helpers/ -> tests/ -> project root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.utils.data

# Actual production functions — no mocks
from cloud_train import (
    TaraNativeTokenizer,
    TaraStreamingJsonlDataset,
    collate_fn,
    save_checkpoint,
    try_resume_from_checkpoint,
    TaraForCausalLM,
)

# Tiny model config — vocab_size=64 so safe IDs are 0..63.
# Dataset records use chars chr(32)..chr(63) which map to IDs 32..63.
_CONFIG = {
    "vocab_size": 64,
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "max_position_embeddings": 512,
    "rms_norm_eps": 1e-5,
    "rope_theta": 10000.0,
    "tie_word_embeddings": False,
}


def _write_tokenizer(path: str) -> None:
    """Write a tokenizer that maps chr(0)=<|unk|>, chr(32)..'?' (32..63) → safe IDs."""
    vocab = {chr(i): i for i in range(32, 64)}
    vocab["<|unk|>"] = 0
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"vocab": vocab}, f)


def _make_scaler():
    try:
        return torch.amp.GradScaler("cpu", enabled=False)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=False)


def _cursors_from_batches(accum_batches):
    """Return all (shard, line, chunk) tuples committed in the accumulation window."""
    units = []
    for (_inp, _lbl, cursors) in accum_batches:
        for c in cursors:
            units.append({
                "shard": c[0],
                "line": int(c[1]) if len(c) > 1 else 0,
                "chunk": int(c[3]) if len(c) > 3 else 0,
            })
    return units


def main(argv=None):
    p = argparse.ArgumentParser(description="Phase 2.1 integration test training helper")
    p.add_argument("--output-dir", required=True,
                   help="Checkpoint output directory (passed to save_checkpoint as output_dir)")
    p.add_argument("--shard-path", required=True,
                   help="Absolute path to the synthetic JSONL shard")
    p.add_argument("--max-steps", type=int, default=2,
                   help="Hard cap on successful optimizer updates")
    p.add_argument("--grad-accum", type=int, default=1,
                   help="Gradient accumulation window size")
    p.add_argument("--max-seq-len", type=int, default=8,
                   help="Maximum sequence length per chunk")
    p.add_argument("--log-file", required=True,
                   help="Path to write the committed-unit JSON log")
    p.add_argument("--resume", action="store_true",
                   help="Attempt to resume from a previous checkpoint in output-dir")
    p.add_argument("--crash-before-commit-step", type=int, default=-1,
                   help="After optimizer step N succeeds, exit(2) BEFORE saving checkpoint")
    p.add_argument("--crash-after-commit-step", type=int, default=-1,
                   help="After saving checkpoint at step N, exit(3) immediately after")
    args = p.parse_args(argv)

    os.makedirs(args.output_dir, exist_ok=True)

    tok_path = os.path.join(args.output_dir, "tokenizer.json")
    _write_tokenizer(tok_path)

    config = dict(_CONFIG)
    model = TaraForCausalLM(config).cpu()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = _make_scaler()

    start_step = 0
    start_epoch = 0
    best_loss = float("inf")
    resumed_cursor = None

    if args.resume:
        start_step, start_epoch, best_loss, _, resumed_cursor = \
            try_resume_from_checkpoint(args.output_dir, model, optimizer, scaler)

    resume_shard = resumed_cursor.get("active_shard") if resumed_cursor else None
    resume_line = resumed_cursor.get("active_shard_line_offset", -1) if resumed_cursor else -1
    resume_byte = resumed_cursor.get("active_shard_byte_offset", 0) if resumed_cursor else 0
    resume_chunk = resumed_cursor.get("active_chunk_idx", 0) if resumed_cursor else 0

    dataset = TaraStreamingJsonlDataset(
        file_paths=[args.shard_path],
        tokenizer=TaraNativeTokenizer(json.load(open(tok_path))["vocab"]),
        max_seq_len=args.max_seq_len,
        resume_shard=resume_shard,
        resume_line_offset=resume_line,
        resume_byte_offset=resume_byte,
        resume_chunk_idx=resume_chunk,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, collate_fn=collate_fn,
    )

    log_entries = []  # one entry per durably committed optimizer step
    step = start_step
    current_cursor = resumed_cursor
    accum_batches = []
    accum_loss = 0.0

    def _flush_log():
        """Write the current log_entries to disk so crash tests can read partial logs."""
        with open(args.log_file, "w", encoding="utf-8") as _f:
            json.dump(log_entries, _f, indent=2)

    model.train()
    optimizer.zero_grad(set_to_none=True)

    # Single epoch — bounded synthetic test
    for batch_idx, (inputs, labels, cursors) in enumerate(dataloader):
        inputs = inputs.cpu()
        labels = labels.cpu()

        _, loss, _, _ = model(inputs, labels=labels)
        (loss / args.grad_accum).backward()
        accum_loss += loss.item()
        accum_batches.append((inputs, labels, cursors))

        if len(accum_batches) >= args.grad_accum:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            gn = float(grad_norm)

            if math.isnan(gn) or math.isinf(gn):
                # Non-finite: discard window, cursor unchanged
                print(f"[Helper] Non-finite gradient at batch {batch_idx} — window discarded",
                      flush=True)
                optimizer.zero_grad(set_to_none=True)
                accum_batches.clear()
                accum_loss = 0.0
                continue

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            committed_units = _cursors_from_batches(accum_batches)

            last_c = accum_batches[-1][2][-1]
            c_chunk_last = int(last_c[3]) if len(last_c) > 3 else 0
            current_cursor = {
                "active_shard": last_c[0],
                "active_shard_line_offset": int(last_c[1]) if len(last_c) > 1 else 0,
                "active_shard_byte_offset": int(last_c[2]) if len(last_c) > 2 else 0,
                "active_chunk_idx": c_chunk_last + 1,  # next to yield
                "total_samples_trained": step,
                "samples_consumed": step,
                "completed_shards": [],
            }

            cur_loss = accum_loss / args.grad_accum
            accum_batches.clear()
            accum_loss = 0.0

            # ── Fault injection: crash BEFORE checkpoint is saved ──
            if step == args.crash_before_commit_step:
                print(f"[Helper] Intentional pre-checkpoint crash at step {step}", flush=True)
                sys.exit(2)

            save_checkpoint(
                model, optimizer, scaler, config, tok_path, args.output_dir,
                step=step, epoch=start_epoch, loss=cur_loss, best_loss=min(best_loss, cur_loss),
                stage_id="integration_test", is_best=False, is_completed=False,
                data_cursor=current_cursor,
            )

            log_entries.append({
                "step": step,
                "committed": True,
                "units": committed_units,
                "cursor": current_cursor,
            })
            # Flush log after every checkpoint so crash tests can read partial logs
            _flush_log()

            # ── Fault injection: crash AFTER checkpoint is saved ──
            if step == args.crash_after_commit_step:
                print(f"[Helper] Intentional post-checkpoint crash at step {step}", flush=True)
                sys.exit(3)

            if step >= args.max_steps:
                break

    # Partial flush at epoch end with correct 1/partial_count normalisation
    if accum_batches and step < args.max_steps:
        partial_count = len(accum_batches)
        optimizer.zero_grad(set_to_none=True)
        partial_loss = 0.0
        for (p_inp, p_lbl, _) in accum_batches:
            p_inp = p_inp.cpu()
            p_lbl = p_lbl.cpu()
            _, p_loss, _, _ = model(p_inp, labels=p_lbl)
            (p_loss / partial_count).backward()
            partial_loss += p_loss.item()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        gn = float(grad_norm)

        if not (math.isnan(gn) or math.isinf(gn)):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            committed_units = _cursors_from_batches(accum_batches)
            last_c = accum_batches[-1][2][-1]
            c_chunk_last = int(last_c[3]) if len(last_c) > 3 else 0
            current_cursor = {
                "active_shard": last_c[0],
                "active_shard_line_offset": int(last_c[1]) if len(last_c) > 1 else 0,
                "active_shard_byte_offset": int(last_c[2]) if len(last_c) > 2 else 0,
                "active_chunk_idx": c_chunk_last + 1,
                "total_samples_trained": step,
                "samples_consumed": step,
                "completed_shards": [],
            }

            save_checkpoint(
                model, optimizer, scaler, config, tok_path, args.output_dir,
                step=step, epoch=start_epoch, loss=partial_loss / partial_count,
                best_loss=float("inf"), stage_id="integration_test",
                is_best=False, is_completed=False, data_cursor=current_cursor,
            )
            log_entries.append({
                "step": step,
                "committed": True,
                "units": committed_units,
                "cursor": current_cursor,
            })
            # Flush log after partial-flush checkpoint
            _flush_log()
        else:
            print("[Helper] Partial flush: non-finite gradient — window discarded", flush=True)

        accum_batches.clear()

    # Final log flush (also handles the case where no crash occurred and no partial flush)
    _flush_log()

    print(f"[Helper] Done. Committed {step - start_step} step(s). Log: {args.log_file}",
          flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
