# TARA Phase 2.1 — End-to-End Recovery Acceptance Report

**Branch:** `phase1-data-safety-fixes`
**Report date:** 2026-10-20
**Prepared from:** commit `de2bee9` (working tree clean)

---

## Branch commit history (Phase 2.1 additions)

```
de2bee9  Phase 2.1: Add subprocess integration tests and Gate 1-4 acceptance tests; 19/19 pass, 85/85 total
efbc6d8  Phase 2.1: Fix partial flush gradient normalisation (1/partial_count, not 1/grad_accum)
4e58d2e  Add TARA_PHASE2_REPORT.md — 66/66 tests pass, 6 fixes documented, risks stated
ca7deb5  Phase 2: Fix cursor write (active_chunk_idx = next to yield), byte-seek guard, partial flush always executed, non-finite grad logged, legacy cursor restore; 66/66 tests pass
aea9202  Add TARA_PHASE1_1_ACCEPTANCE_REPORT.md — all 6 gates PASS, 49/49 tests
23a7b9f  Phase 1.1: Fix resume line-skip boundary, add versioned-dir recovery, buffer-size guard, best-ckpt COMMITTED marker, retention policy; 49/49 tests pass
b5a1587  Add TARA_PHASE1_IMPLEMENTATION_REPORT.md
37525a4  Phase 1: Crash-safe staged checkpoints, deterministic token chunking, deferred shard completion
```

---

## Working-tree status at report time

```
On branch phase1-data-safety-fixes
nothing to commit, working tree clean
```

---

## Phase 2.1 source changes

### Commit `efbc6d8` — `cloud_train.py` (partial flush gradient normalisation fix)

**Root cause (identified in Phase 2.1 audit):**  
The partial accumulation flush at epoch end reused gradients accumulated with
`loss / grad_accum` per batch. When the window is partial (K < grad_accum
batches), the effective gradient magnitude was `K / grad_accum` of a full
window — systematically under-scaled by the number of remaining batches.

**Fix applied:**  
Before the partial flush `optimizer.step()`, zero gradients and re-run
forward + backward for each batch in the partial window using
`loss / partial_count` as the divisor, where `partial_count = len(accum_batches)`.
This produces gradient magnitudes equivalent to a full window of that size.

**Location:** `cloud_train.py`, partial-flush block in `main()`.

---

### Commit `de2bee9` — test infrastructure additions

| File | Purpose |
|------|---------|
| `tests/helpers/__init__.py` | Package marker (empty) |
| `tests/helpers/phase2_1_train_helper.py` | Bounded subprocess training helper |
| `tests/test_phase2_1.py` | Gate 1–4 acceptance test suite (19 tests) |

**Helper design note:**  
The helper writes `log_entries` to disk via `_flush_log()` after every
successful checkpoint commit, not only at normal exit. This ensures crash
tests (exit code 2 / exit code 3) leave a partial but correct log that the
test suite can inspect.

---

## Test commands executed

### Phase 2.1 suite (19 tests)
```
C:\Python314\python.exe -m pytest tests/test_phase2_1.py -v --tb=short
```

### Full regression suite (85 tests)
```
C:\Python314\python.exe -m pytest tests/test_phase1.py tests/test_phase1_1.py tests/test_phase2.py tests/test_phase2_1.py -v --tb=short
```

---

## Full regression result

```
platform win32 -- Python 3.14.7, pytest-9.1.1, pluggy-1.7.0
collected 85 items

[Phase 1  — 17 tests] ALL PASSED
[Phase 1.1 — 27 tests] ALL PASSED
[Phase 2  — 22 tests] ALL PASSED
[Phase 2.1 — 19 tests] ALL PASSED

85 passed, 7 subtests passed in 146.23s (0:02:26)
```

---

## Gate-by-gate results

### Gate 1 — Fresh-process resume integration  PASS

**What was tested:**  
Real subprocess A trains N steps, commits a versioned checkpoint, and exits.
Subprocess B starts from a fresh Python process, discovers and restores the
checkpoint, and continues training. All assertions use the actual
`save_checkpoint` / `try_resume_from_checkpoint` production code paths.

| Test | Description | Result |
|------|-------------|--------|
| `test_h_int1_normal_resume_no_duplicate_no_omission` | B's committed units do not overlap A's; no unit skipped | **PASS** |
| `test_h_int2_resumed_step_counter_matches_checkpoint` | B's first optimizer step is labelled `start_step + 1` from checkpoint | **PASS** |
| `test_h_int3_crash_before_checkpoint_recovers_from_prior` | A crashes at step 2 before checkpoint write; B resumes from step 1 cursor | **PASS** |
| `test_h_int4_grad_accum_gt1_resume_continues_from_cursor` | With `grad_accum=2`, B resumes exactly at the first uncommitted chunk | **PASS** |

**Scope and limitation explicitly acknowledged:**  
Gate 1 proves that **no durably committed training unit (one that reached a
successful `save_checkpoint` call) is re-applied to an optimizer update in
the resumed session**. It does not prove zero-duplicate training across all
possible intermediate failure modes that do not produce a committed checkpoint
(see Remaining Risks R1).

---

### Gate 2 — `max_steps` hard cap and partial gradient accumulation  PASS

**What was tested:**  
`max_steps` is a hard cap on successful optimizer updates. Partial
accumulation windows at epoch end are flushed with correct `1/partial_count`
gradient normalisation. No optimizer update may exceed the cap.

| Test | Description | Result |
|------|-------------|--------|
| `test_h_g2_1_full_windows_respect_max_steps_hard_cap` | Full windows stop exactly at `max_steps` | **PASS** |
| `test_h_g2_2_partial_window_at_epoch_end_completes_last_step` | Partial window at epoch end produces one additional step | **PASS** |
| `test_h_g2_3_max_steps_1_partial_remainder_not_flushed` | When `max_steps=1` is hit inside the main loop, `accum_batches` is empty on exit — partial flush produces nothing | **PASS** |
| `test_h_g2_4_partial_flush_gradient_normalised_by_partial_count` | Gradient norm ratio `partial_count/grad_accum` ≈ 2.0 when `grad_accum=4`, `partial_count=2` | **PASS** |
| `test_h_g2_5_step_counter_never_exceeds_max_steps` (parametrised ×4) | Step count ≤ `max_steps` across four (max_steps, grad_accum, n_docs) combinations | **PASS** |

**`max_steps` hard-cap proof:**  
The `step >= max_steps` break inside the inner loop always exits before
re-entering the accumulation block, leaving `accum_batches` empty at that
point. The partial flush only fires when `accum_batches` is non-empty AND
`step < max_steps`, meaning the partial flush can advance step to at most
`max_steps` — never beyond.

---

### Gate 3 — Non-finite gradient handling  PASS

**What was tested:**  
NaN/Inf gradients are detected on both the main accumulation loop and the
partial flush path, independently of `GradScaler` state (CPU-only, scaler
disabled). Neither optimizer step, committed step counter, nor data cursor
must advance on a skipped update.

| Test | Description | Result |
|------|-------------|--------|
| `test_h_g3_1_nan_gradient_main_loop_step_and_cursor_not_advanced` | Poisoned gradients in main loop → step and cursor frozen | **PASS** |
| `test_h_g3_2_nan_gradient_partial_flush_step_and_cursor_not_advanced` | Poisoned gradients in partial flush → step and cursor frozen, explicit log emitted | **PASS** |
| `test_h_g3_3_training_continues_after_skipped_step` | Round 1: NaN skip (step stays 0); Round 2: clean update (step becomes 1) | **PASS** |
| `test_h_g3_4_non_finite_detected_without_grad_scaler` | Inf gradient detected via grad norm check alone (no scaler), step not advanced | **PASS** |

**CPU-only note:**  
All Gate 3 tests run on CPU with `GradScaler(enabled=False)`. Scaler's
`unscale_()` / `update()` path is not exercised. The grad-norm check
(`math.isnan(gn) or math.isinf(gn)`) provides the non-finite detection
independently of scaler state.

---

### Gate 4 — Cursor semantics  PASS

**What was tested:**  
`active_chunk_idx` consistently means "first uncommitted chunk index for the
resume line". The dataset iterator correctly skips committed chunks and starts
training from the exact next uncommitted unit.

| Test | Description | Result |
|------|-------------|--------|
| `test_h_g4_1_resume_chunk_zero_skips_entire_resume_line` | `active_chunk_idx=0` means the whole resume line is committed; iterator skips it | **PASS** |
| `test_h_g4_2_resume_from_middle_chunk_skips_committed_chunks` | `active_chunk_idx=N` skips chunks 0..N-1; yields from chunk N | **PASS** |
| `test_h_g4_3_resume_past_final_chunk_yields_nothing_for_resume_line` | Out-of-range `active_chunk_idx` yields nothing for resume line | **PASS** |
| `test_h_g4_4_completed_shard_skipped_via_ledger` | Shard marked complete in ledger is fully skipped | **PASS** |
| `test_h_g4_5_stale_byte_offset_falls_back_to_sequential_scan` | Out-of-range byte offset triggers sequential line scan fallback | **PASS** |
| `test_h_g4_6_valid_byte_with_nonzero_chunk_skips_byte_seek` | Valid byte offset with `active_chunk_idx > 0` skips byte seek (sequential scan used) | **PASS** |

---

## Summary

| Gate | Tests | Result |
|------|-------|--------|
| Gate 1 — Fresh-process resume integration | 4 | **PASS** |
| Gate 2 — `max_steps` hard cap and partial flush | 5 | **PASS** |
| Gate 3 — Non-finite gradient handling | 4 | **PASS** |
| Gate 4 — Cursor semantics | 6 | **PASS** |
| **Total** | **19** | **19 PASS / 0 FAIL** |
| Full regression (Phase 1 + 1.1 + 2 + 2.1) | 85 | **85 PASS / 0 FAIL** |

---

## Remaining Risks

**R1 — Exactly-once training is not proven for all failure modes.**  
Gate 1 proves no durably committed unit is re-applied in the resumed session.
It does not cover failures that occur while the CPU is executing user code
between two checkpoints without producing a committed checkpoint at all (e.g.,
OOM kill after optimizer step but before `save_checkpoint` completes staging).
In that scenario the uncommitted window is re-trained from the previous
committed cursor — at-least-once, not exactly-once.

**R2 — Byte-offset resume validity is dataset-content-dependent.**  
`resume_byte_offset` is only valid when the dataset file content is identical
to what was present during the session that wrote the checkpoint. Content
changes (file replacement, re-download) invalidate the offset silently; the
out-of-range guard catches only the size-reduction case. A content-hash guard
is not yet implemented.

**R3 — GradScaler path not exercised on CPU.**  
All tests run CPU-only (`GradScaler(enabled=False)`). The scaler
`unscale_()` / `step()` / `update()` path is not tested here. CUDA-enabled
training must be validated separately before production use.

**R4 — Single-shard, single-epoch scope.**  
The integration tests use a single JSONL shard and a single epoch. Cross-shard
boundary resume and multi-epoch cursor continuity are not covered by these
tests.

**R5 — Partial flush `max_steps` interaction with long tail.**  
If a partial flush raises step from `max_steps - 1` to `max_steps`, the loop
exits correctly. A partial flush cannot raise step beyond `max_steps` (proved
by H_G2_3 and H_G2_5). However, if the partial flush itself encounters a
non-finite gradient, training stops at `max_steps - 1` — the test suite
confirms step is frozen, but the training run ends one short without a
warning. This is correct behaviour but warrants explicit logging in
production.

**R6 (carried from Phase 2 report) — Ledger and checkpoint can diverge in
distributed or multi-process settings.**  
All tests are single-process or two-process sequential. Concurrent access to
the ledger file or checkpoint directory is not tested.

---

## What is NOT claimed

- This report does NOT claim production-readiness.
- This report does NOT claim zero-duplicate training across all failure modes.
- This report does NOT claim CUDA correctness.
- This report does NOT authorise merging to `main` or starting production training.

---

*End of TARA Phase 2.1 Acceptance Report*
