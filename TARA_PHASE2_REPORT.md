# TARA Phase 2 — Training Hygiene and Resume Audit Report

**Branch:** `phase1-data-safety-fixes`
**Starting commit (Phase 1.1 head):** `aea9202`
**Phase 2 implementation commit:** `ca7deb5`
**Report commit:** (see below — committed immediately after this file)
**Date:** 2026-10-20
**Python:** 3.14.7 | **PyTorch:** 2.14.0+cpu | **pytest:** 9.1.1
**Full test result: 66 passed, 3 subtests passed — 0 failures**

---

## 1. Starting State

Phase 1.1 was accepted at commit `aea9202` with 49/49 tests passing across:
- `tests/test_phase1.py` — 22 D-series tests
- `tests/test_phase1_1.py` — 27 G-series tests

Branch `phase1-data-safety-fixes` was clean, no uncommitted changes.
`main` remains at `b2ec7cf` — untouched throughout.

---

## 2. Phase 2 Issues Investigated and Fixed

### P2-A — Cursor write stored last-committed chunk instead of next-to-yield chunk

**Root cause (in `cloud_train.py`):**

The training loop extracted `c_chunk = last_sample_cursor[3]` (the `chunk_idx` of the last sample that entered the successful optimizer step) and wrote it directly as `"active_chunk_idx": c_chunk` in the checkpoint cursor.

The dataset iterator interprets `active_chunk_idx` as the exclusive lower bound:
```python
if chunk_idx < target_resume_chunk:
    continue   # skip this chunk — it is committed
```

With `c_chunk = 1` stored as `active_chunk_idx`, the iterator skips `chunk_idx < 1` (i.e., only chunk 0) and re-yields chunk 1 — which was the last sample in the committed optimizer step. **Chunk 1 was trained twice.**

**Fix (commit `ca7deb5`, functions: main training loop cursor write + partial flush cursor write):**

```python
# Before
c_chunk = last_sample_cursor[3] if len(last_sample_cursor) > 3 else 0
current_cursor = { ..., "active_chunk_idx": c_chunk, ... }

# After
c_chunk_last = last_sample_cursor[3] if len(last_sample_cursor) > 3 else 0
c_chunk_next = c_chunk_last + 1   # first uncommitted chunk
current_cursor = { ..., "active_chunk_idx": c_chunk_next, ... }
```

The same fix was applied to the end-of-epoch partial flush cursor write.

**Established guarantee (test H5):**
`test_cursor_stores_next_chunk_not_last_committed` and `test_cursor_active_chunk_idx_stored_in_checkpoint` prove that `active_chunk_idx = c_chunk_last + 1` is written and faithfully round-tripped through `save_checkpoint` → `try_resume_from_checkpoint`.

---

### P2-B — Byte-seek path silently skipped uncommitted chunks of the resume line

**Root cause:**

The stored `resume_byte_offset` is captured as `f.tell()` *after* reading `target_resume_line`, positioning the file pointer at the *start* of the next line. The byte-seek path set `start_line_idx = target_resume_line + 1` — correct when the whole resume line is committed (`active_chunk_idx == 0`), but wrong when `active_chunk_idx > 0`: the resume line still has uncommitted chunks (indices `active_chunk_idx` onward) that must be replayed.

**Fix (commit `ca7deb5`, function: `TaraStreamingJsonlDataset.__iter__`):**

The byte-seek optimisation is now gated on `target_resume_chunk == 0`:
```python
if is_resume_target and target_resume_byte > 0 and target_resume_chunk == 0:
    # Whole resume line committed — safe to byte-seek to start of next line
    file_end = f.seek(0, 2)
    if 0 < target_resume_byte <= file_end:
        f.seek(target_resume_byte)
        start_line_idx = max(0, target_resume_line + 1)
    else:
        f.seek(0)   # out-of-range fallback
        start_line_idx = 0
```

When `target_resume_chunk > 0`, the byte-seek is skipped entirely. The sequential scan fast-forward path handles it: the resume line is re-read, and the per-chunk filter `chunk_idx < target_resume_chunk` skips the committed chunks.

**Established guarantee (test H3):**
`test_byte_seek_skipped_when_resume_chunk_nonzero` proves that with `resume_chunk_idx=2` and a valid `resume_byte_offset`, only chunk 2 of the resume line is yielded (chunks 0,1 not re-yielded; lines after resume yielded in full).

---

### P2-C — Byte-seek validity guard (out-of-range offset)

**Root cause:** The previous byte-seek code performed `f.seek(target_resume_byte)` without checking whether the offset was within the current file bounds. If the dataset file was replaced or truncated after the checkpoint was written, the seek could land at an arbitrary position.

**Fix (commit `ca7deb5`, same function):** Before seeking, the file end position is queried with `f.seek(0, 2)`. If `target_resume_byte` exceeds the file end (or is zero), the fallback path `f.seek(0); start_line_idx = 0` is used, causing a clean sequential scan from the start of the shard.

**Established guarantee (test H3):**
`test_out_of_range_byte_offset_falls_back_to_sequential` proves that an offset of `file_size + 9999` produces output identical to a fresh iteration.

---

### P2-D — End-of-epoch partial flush skipped when `step >= max_steps`

**Root cause:**

```python
if accum_batches and step < max_steps:   # old guard
```

When `step >= max_steps` at epoch end, any remaining accumulated gradient work was silently abandoned. The model parameters had already received the gradient from those batches but the corresponding optimizer update was never applied; the committed cursor never advanced; the samples were neither counted as trained nor marked as skippable on resume.

**Fix (commit `ca7deb5`, function: `main()` end-of-epoch block):**

```python
if accum_batches:   # always flush — no max_steps guard
```

The partial window is now flushed at every epoch end regardless of step count. The `step >= max_steps` break that halts the outer epoch loop remains in place so training still terminates at the correct step boundary.

**Established guarantee (test H4):**
`test_partial_flush_executes_regardless_of_max_steps` simulates the partial flush block with `max_steps == step` and confirms that `step` advances by 1 and `current_cursor` is set.

---

### P2-E — Non-finite gradient in partial flush: silent discard

**Root cause:** The partial flush did not log anything when `is_non_finite_grad and not scaler.is_enabled()` caused the optimizer step to be skipped. The training work accumulated in the partial window was quietly dropped.

**Fix (commit `ca7deb5`, function: `main()` end-of-epoch block):**

An explicit print is now emitted on skip:
```python
print(
    f"[Partial Window Non-Finite Gradient] Inf/NaN gradient detected in "
    f"end-of-epoch partial accumulation window ({partial_count} batches). "
    f"Optimizer step skipped; partial window discarded. Committed step "
    f"and cursor are unchanged.",
    flush=True,
)
```

The same explicit message pattern applies to the AMP scale-reduction case.

**Established guarantee (test H4):**
`test_non_finite_gradient_partial_flush_logs_and_does_not_advance_step` poisons gradients with NaN, runs the partial flush block, and verifies: (a) `step` does not advance, (b) the log output contains `"Non-Finite Gradient"`.

---

### P2-F — Legacy cursor fallback missing `active_chunk_idx`

**Root cause:** The fallback path in `try_resume_from_checkpoint` that reconstructs a cursor from flat `checkpoint_state.json` fields (no `data_cursor` sub-key) did not include `"active_chunk_idx"`. Checkpoints written before Phase 2 would resume with `active_chunk_idx = 0` (whole resume line skipped) regardless of the actual committed chunk.

**Fix (commit `ca7deb5`, function: `try_resume_from_checkpoint`):**

```python
# Flat legacy layout fallback — now includes active_chunk_idx
best_cand_cursor = {
    "active_shard": s_data.get("active_shard"),
    "active_shard_line_offset": s_data.get("active_shard_offset", 0),
    "active_shard_byte_offset": s_data.get("active_shard_byte_offset", 0),
    "active_chunk_idx": s_data.get("active_chunk_idx", 0),   # ← added
    ...
}
# Modern data_cursor layout — now backfills active_chunk_idx if absent
if "active_chunk_idx" not in best_cand_cursor:
    best_cand_cursor["active_chunk_idx"] = s_data.get("active_chunk_idx", 0)
```

**Established guarantee (test H6):**
`test_legacy_flat_cursor_restores_active_chunk_idx` writes a checkpoint with no `data_cursor` key but with `active_chunk_idx=3` in the flat layout, calls `try_resume_from_checkpoint`, and asserts the returned cursor has `active_chunk_idx == 3`.

---

### resume_chunk_idx Semantics — Documented and Regression-Tested

**Clarification (no code change required):**

`active_chunk_idx` in the checkpoint cursor (and `resume_chunk_idx` in the dataset constructor) is the index of the **first uncommitted chunk** within the resume line — i.e., the exclusive lower bound for the per-chunk skip filter `chunk_idx < target_resume_chunk`.

| Stored value | Meaning | Iterator behaviour |
|---|---|---|
| `0` | Entire resume line committed | Skip the whole line (`line_idx == target_resume_line and chunk_idx == 0 → continue`) |
| `N > 0` | Chunks 0..N-1 are committed | Skip `chunk_idx < N`; yield from chunk N onward |
| `N == total_chunks` | All chunks of line committed | Skip filter fires for every chunk; line yields nothing |

The training loop now writes `c_chunk_last + 1` which maps directly onto this contract: if the last committed chunk was chunk 1, `active_chunk_idx = 2`, so the iterator skips chunks 0,1 and starts from chunk 2.

**Established guarantees (tests H1):**
- `test_chunk_idx_0_skips_entire_resume_line` — `active_chunk_idx=0` skips whole line
- `test_chunk_idx_1_skips_only_chunk_0` — `active_chunk_idx=1` yields chunks 1,2
- `test_chunk_idx_2_skips_chunks_0_and_1` ← **canonical regression per Phase 2 audit** — `active_chunk_idx=2` skips chunks 0,1, yields chunk 2 only
- `test_chunk_idx_beyond_last_chunk_yields_nothing_for_resume_line`
- `test_no_chunk_appears_before_resume_line` (parametrised over 3 resume_chunk values)

---

## 3. Files Changed

| File | Change |
|---|---|
| `cloud_train.py` | `TaraStreamingJsonlDataset.__iter__`: byte-seek guard + skip condition; main training loop cursor write: `c_chunk + 1`; end-of-epoch partial flush: removed `step < max_steps` guard, added explicit non-finite log; `try_resume_from_checkpoint`: legacy and modern cursor fallback now include `active_chunk_idx` |
| `tests/test_phase2.py` | New — 17 tests across 6 test classes (H1–H6), 3 parametrised subtests |

Net diff summary: `+930 lines, -18 lines` across 2 files.

---

## 4. Complete Test Results

```
Command: C:\Python314\python.exe -m pytest tests/test_phase1.py tests/test_phase1_1.py tests/test_phase2.py -v

Platform: win32 — Python 3.14.7, pytest 9.1.1, PyTorch 2.14.0+cpu (CPU only)

tests/test_phase1.py          22 PASSED
tests/test_phase1_1.py        27 PASSED
tests/test_phase2.py          17 PASSED  (+ 3 subtests PASSED)
─────────────────────────────────────────
Total: 66 passed, 3 subtests passed in 21.81s
```

All prior 49 Phase 1 / Phase 1.1 tests continue to pass without modification.

---

## 5. Commit History on `phase1-data-safety-fixes`

| Commit | Content |
|---|---|
| `b2ec7cf` | `main` baseline |
| `37525a4` | Phase 1 core: staged checkpoints, token chunking, deferred shard completion |
| `b5a1587` | `TARA_PHASE1_IMPLEMENTATION_REPORT.md` |
| `23a7b9f` | Phase 1.1: resume boundary fix, versioned-dir recovery, buffer-size guard, COMMITTED marker, retention |
| `aea9202` | `TARA_PHASE1_1_ACCEPTANCE_REPORT.md` |
| `ca7deb5` | **Phase 2: cursor write, byte-seek guard, partial flush, non-finite log, legacy restore; 66/66 tests** |

---

## 6. Remaining Risks and Known Limitations

### R1 — `active_chunk_idx` overflow at shard boundary
When the last optimizer step of a shard commits the final chunk of the last line (say chunk K), the cursor stores `active_chunk_idx = K + 1`. On resume, the iterator skips all K+1 chunks of that line (correctly, since all are committed) and moves to the next shard or line. The shard itself will be in `completed_shards` via `commit_shard_completion`, so this scenario should never be reached with a real shard that completed. However, if `commit_shard_completion` was not called (e.g., crash between optimizer step and checkpoint commit), the resume will re-stream the last chunk with `active_chunk_idx = K + 1` hitting no chunks — producing a correct (empty) re-stream of that line, then continuing normally. The training work for that chunk is lost (not duplicate-trained). This is the same risk documented in Phase 1.1 R1.

### R2 — Byte-offset staleness after file replacement
The byte-seek path now guards against `target_resume_byte > file_size`. It does **not** verify that the content at the seek position actually corresponds to `target_resume_line + 1` (e.g., the file was replaced with a same-size file). Content verification would require re-tokenizing the line at the seek position, which is expensive and would defeat the purpose of the optimisation. The existing guard is sufficient for the FUSE/Google Drive semantics where file replacement is not atomic and a byte-seek to a stale offset would land mid-line or in whitespace, causing `json.loads` to fail gracefully and the line to be skipped. No test establishes a guarantee for the content-replacement scenario.

### R3 — Partial flush non-finite gradient: training work is dropped
When the partial flush window is skipped due to a non-finite gradient, the accumulated gradient batches are discarded. The committed model state and cursor are preserved correctly (no step advance, no cursor advance), but the training samples that produced the gradient are not retried — they are simply re-streamed from the cursor position on the next run. This is the correct fail-safe behaviour for model parameter integrity, but it means those samples will be trained again on resume (not duplicate-trained relative to the committed model; they were never committed). This is documented by the explicit log message now emitted.

### R4 — Retry loop absent from partial flush
The main accumulation step has a retry loop (up to 5 retries on non-finite gradients). The partial flush does not. A single non-finite gradient in the partial window discards the entire partial window immediately. This is conservative and correct for safety but means the partial window does not benefit from the retry logic. Aligning the retry behaviour would require refactoring the retry loop into a shared helper — deferred to a future clean-up.

### R5 — `resume_byte_offset` not versioned with dataset content hash
The byte offset in the cursor is valid only for the exact version of the shard file that was active when the checkpoint was written. No dataset content hash is stored alongside the byte offset. If the shard file is modified between checkpoint write and resume, the byte-seek will land at an incorrect position. The out-of-range guard catches truncations but not same-size modifications. This is a known structural limitation documented in the Phase 1 audit.

### R6 — No end-to-end fresh-process resume simulation
All Phase 2 tests operate within a single process. The guarantee that `try_resume_from_checkpoint` faithfully restores `active_chunk_idx` and that the dataset iterator uses it correctly is established by separate unit tests (H5, H6), not by an integrated test that spawns a fresh process, runs N steps, checkpoints, exits, and then spawns another process to resume and verify exact training-unit IDs. A true fresh-process integration test is the remaining gap for a zero-duplicate-training guarantee.

---

## 7. What Phase 2 Does NOT Claim

- **Zero duplicate training across all failure modes.** The tests establish that the cursor write is correct and the iterator semantics are correct independently. An integrated end-to-end proof with exact training-unit ID tracking across a process boundary has not been executed.
- **Byte-offset validity after content change.** Only out-of-range (truncation) is guarded. Same-size file replacement is not detected.
- **Retry parity between main loop and partial flush.** The partial flush uses a single-attempt policy; the main loop retries up to 5 times.
- **Production readiness.** TARA is in active development and is not ready for release.
