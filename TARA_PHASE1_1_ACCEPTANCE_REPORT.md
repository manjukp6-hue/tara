# TARA Phase 1.1 Acceptance Gate Report

**Branch:** `phase1-data-safety-fixes`
**Report commit:** `23a7b9f`
**Report date:** 2026-10-20
**Python:** 3.14.7 | **PyTorch:** 2.14.0+cpu | **pytest:** 9.1.1

---

## Commit History (branch vs. `main` at `b2ec7cf`)

| Commit | Message |
|--------|---------|
| `23a7b9f` | Phase 1.1: Fix resume line-skip boundary, add versioned-dir recovery, buffer-size guard, best-ckpt COMMITTED marker, retention policy; 49/49 tests pass |
| `b5a1587` | Add TARA_PHASE1_IMPLEMENTATION_REPORT.md |
| `37525a4` | Phase 1: Crash-safe staged checkpoints, deterministic token chunking, deferred shard completion |

---

## Test Results Summary

```
49 passed in 20.24s  (22 original D-series + 27 new G-series)
0 failed
Platform: win32 — Python 3.14.7, pytest-9.1.1
```

---

## Original 22 Tests (D-series) — All Pass

| # | Test | Assertions covered |
|---|------|--------------------|
| D1.1 | `test_failure_during_model_write_preserves_prior_checkpoint` | Staging fail → output_dir byte-for-byte unchanged; integrity still passes |
| D1.2 | `test_failure_during_optimizer_write_preserves_prior_checkpoint` | Optimizer crash → step in checkpoint_state.json remains at prior value |
| D1.3 | `test_failure_during_manifest_write_preserves_prior_checkpoint` | manifest crash → output_dir files unchanged by SHA |
| D1.4 | `test_incomplete_staging_dir_not_accepted_as_valid_checkpoint` | Staging dir without COMMITTED fails `verify_checkpoint_integrity` |
| D2.1 | `test_model_weights_restored_exactly` | Fresh process reloads model weights matching original (atol=1e-6) |
| D2.2 | `test_data_cursor_restored_exactly` | Cursor fields (shard, line_offset, byte_offset, completed_shards) restored exactly |
| D3.1 | `test_dataset_iterator_covers_all_tokens_via_streaming` | End-to-end: all tokens from a long doc appear exactly once as targets |
| D3.2 | `test_final_short_chunk_policy_discard_below_4` | 13-token doc → 1 chunk; 14-token doc → 2 chunks |
| D3.3 | `test_input_output_document_output_tokens_covered_exactly_once` | Output tokens appear as targets exactly once; none lost, none duplicated |
| D3.4 | `test_pure_text_document_all_tokens_covered` | Pure-text target tokens equal expected sequence (minus discarded final short chunk) |
| D4.1 | `test_shard_not_completed_in_ledger_after_streaming_eof` | Shard in `_pending_shard_completions` not in `ledger.completed_shards` after EOF |
| D4.2 | `test_shard_file_not_evicted_before_checkpoint_commit` | File exists after EOF; deleted only after `commit_shard_completion()` |
| D4.3 | `test_recoverable_source_preserved_if_crash_before_checkpoint` | Shard not completed; file readable; re-streamable in second dataset instance |
| D5.1 | `test_ledger_reconciled_from_checkpoint_after_ledger_update_crash` | Stale-behind ledger brought to checkpoint cursor by `reconcile_with_checkpoint()` |
| D5.2 | `test_stale_ahead_ledger_evicted_by_reconciliation` | Premature shard completion removed from ledger by reconciliation |
| D6.1 | `test_resume_skips_committed_lines_and_chunks` | Lines 0–2 not yielded; lines 3–4 yielded after `resume_line_offset=2` |
| D7.1 | `test_uncommitted_chunks_reprocessed_after_resume` | All 3 chunks of lines 1 and 2 appear after resume; line 0 absent |
| D8.1 | `test_user_source_file_not_evicted` | Non-managed file survives `commit_shard_completion(managed_cache=False)` |
| D8.2 | `test_managed_cache_file_evicted_after_commit` | Managed file deleted; shard in `ledger.completed_shards` |
| D8.3 | `test_keep_shards_flag_prevents_eviction` | Managed file NOT deleted when `keep_shards=True` |
| CollateFn.1 | `test_6tuple_items_collated_correctly` | 6-tuple items produce correct `(inputs, labels, cursors)` with chunk_idx at position 3 |
| CollateFn.2 | `test_empty_batch_returns_empty_tensors` | Empty batch → zero-numel tensors, empty cursor list |

---

## New Gate Regression Tests (G-series) — All Pass

### Gate G1 — Token / Chunk Completeness — PASS

| Test | Result | Covers |
|------|--------|--------|
| `test_short_doc_under_4_tokens_entirely_discarded` | **PASS** | 3-token doc yields 0 chunks — entire doc discarded by `< 4` policy |
| `test_exact_length_document_produces_one_full_chunk` | **PASS** | Exactly max_seq_len tokens → 1 chunk, all non-masked labels, chunk_idx=0 |
| `test_long_document_all_chunks_covered_no_duplication` | **PASS** | 3×max_seq_len doc → 3 chunks; no duplicate token IDs across chunks |
| `test_all_input_span_chunk_has_all_labels_masked` | **PASS** | Chunk fully inside input span → all labels == -100 (zero loss, still yielded) |
| `test_output_tokens_appear_as_targets_exactly_once` | **PASS** | Multi-chunk input/output doc: each output token is a target exactly once |
| `test_final_short_chunk_exactly_4_tokens_is_yielded` | **PASS** | Boundary: 4-token final chunk is yielded |
| `test_final_short_chunk_3_tokens_is_discarded` | **PASS** | Boundary: 3-token final chunk is discarded |

---

### Gate G2 — Ordering and Cursor Correctness — PASS

| Test | Result | Covers |
|------|--------|--------|
| `test_buffer_size_1_produces_deterministic_order` | **PASS** | Two runs with `buffer_size=1` yield identical sequential item order |
| `test_buffer_size_gt1_with_resume_emits_warning` | **PASS** | `buffer_size=4` + `resume_shard` set → stdout contains "Warning" about non-determinism |
| `test_chunk_idx_in_yielded_cursor_is_correct` | **PASS** | Chunk indices in item tuple increase monotonically: [0, 1, 2] |
| `test_resume_with_chunk_idx_skips_already_committed_chunks` | **PASS** | `resume_line_offset=1, resume_chunk_idx=1`: line 0 absent; (line1,chunk0) absent; (line1,chunk≥1) present; all line2 chunks present |

**Root cause resolved:** `cloud_train.py` line 804 originally used `line_idx <= target_resume_line` which skipped the resume line entirely even when `resume_chunk_idx > 0`. Corrected to a two-stage check:
- `line_idx < target_resume_line` → always skip pre-resume lines
- `line_idx == target_resume_line and resume_chunk_idx == 0` → skip the resume line only when all its chunks are committed

**Source:** [`cloud_train.py:803–816`](file:///C:/Users/Nawaz/Desktop/taracore/cloud_train.py#L803-L816)

---

### Gate G3 — Checkpoint Durability — PASS

| Test | Result | Covers |
|------|--------|--------|
| `test_resume_uses_versioned_dir_via_pointer_not_output_dir` | **PASS** | After save, corrupt model.safetensors in output_dir; resume still succeeds via versioned dir |
| `test_versioned_dir_without_committed_marker_rejected` | **PASS** | Versioned dir without COMMITTED not returned by `_collect_versioned_checkpoint_dirs` |
| `test_committed_versioned_dirs_discovered_newest_first` | **PASS** | 5 committed dirs returned sorted by step descending |
| `test_crash_before_committed_marker_leaves_prior_checkpoint_recoverable` | **PASS** | Uncommitted step-2 dir present; recovery finds step-1 committed dir |

**Root cause resolved:** `try_resume_from_checkpoint` previously read from `checkpoint_dir` (output_dir) directly. A crash mid-copy into output_dir leaves a mutually-inconsistent mix of artifacts. The fix adds:
1. `_collect_versioned_checkpoint_dirs()` — scans for committed versioned dirs, sorted newest-first
2. `latest_checkpoint.json` pointer is the highest-priority candidate
3. Optimizer loaded directly from the verified `best_cand_dir` (immutable versioned dir), not from output_dir

**Source:** [`cloud_train.py:2035–2347`](file:///C:/Users/Nawaz/Desktop/taracore/cloud_train.py#L2035-L2347)

---

### Gate G4 — Checkpoint / Ledger Consistency — PASS

| Test | Result | Covers |
|------|--------|--------|
| `test_pending_completions_lost_on_restart_causes_resream_not_loss` | **PASS** | In-memory pending set lost on restart; shard re-streams with same item count |
| `test_stale_behind_ledger_reconciled_from_checkpoint` | **PASS** | `reconcile_with_checkpoint` restores completed shard, active cursor, committed step |
| `test_stale_ahead_ledger_reconciled_removes_premature_completions` | **PASS** | Premature shard removed; checkpoint-authoritative shard retained |
| `test_shard_completion_only_after_checkpoint_commit` | **PASS** | Pending before `save_checkpoint`; in `ledger.completed_shards` after |

> **Note:** `_pending_shard_completions` is intentionally in-memory only. On restart it is empty. The shard is re-streamed (not lost) — correct at-least-once semantics.

---

### Gate G5 — Best Model Safety — PASS

| Test | Result | Covers |
|------|--------|--------|
| `test_committed_best_checkpoint_has_committed_marker` | **PASS** | `save_best_milestone` writes `COMMITTED` with `is_best=True` after atomic commit |
| `test_best_checkpoint_without_optimizer_excluded_from_resume` | **PASS** | best_checkpoint has no `optimizer_state.pt`; `try_resume_from_checkpoint` returns `was_resumed=False` |
| `test_incomplete_best_checkpoint_rejected_by_integrity_check` | **PASS** | Partial best dir fails `verify_checkpoint_integrity` |
| `test_best_checkpoint_not_selected_over_training_checkpoint` | **PASS** | Training ckpt (step=5) selected over best_checkpoint (step=99, no optimizer) |

**Root cause resolved:** `save_best_milestone` did not write a `COMMITTED` marker. Now writes it after atomic promotion succeeds.

**Source:** [`cloud_train.py:1644–1656`](file:///C:/Users/Nawaz/Desktop/taracore/cloud_train.py#L1644-L1656)

---

### Gate G6 — Storage Retention — PASS

| Test | Result | Covers |
|------|--------|--------|
| `test_retention_keeps_newest_two_deletes_older` | **PASS** | 4 committed dirs (steps 1–4): steps 3,4 remain; steps 1,2 deleted |
| `test_retention_removes_uncommitted_staging_remnants` | **PASS** | Uncommitted staging remnant removed regardless of keep_n |
| `test_retention_never_keeps_fewer_than_2` | **PASS** | `keep_n=1` enforced to minimum 2; step-1 deleted, steps 2,3 retained |
| `test_retention_called_after_save_checkpoint_keeps_two` | **PASS** | 3 sequential saves: ≤2 versioned dirs remain; step-3 present |

**Implementation:** `_cleanup_old_versioned_checkpoints(parent_dir, basename, keep_n=2)` called inside `save_checkpoint` after `latest_checkpoint.json` pointer is published. Enforces minimum of 2 retained checkpoints.

**Source:** [`cloud_train.py:2095–2161`](file:///C:/Users/Nawaz/Desktop/taracore/cloud_train.py#L2095-L2161)

---

## All Source Changes in commit `23a7b9f`

| Change | File:Line | Description |
|--------|-----------|-------------|
| `_collect_versioned_checkpoint_dirs()` new | `cloud_train.py:2035` | Scans parent for COMMITTED versioned dirs, sorted newest-first |
| `_cleanup_old_versioned_checkpoints()` new | `cloud_train.py:2063` | Retention policy: keep_n=2 minimum, delete older + uncommitted |
| `try_resume_from_checkpoint()` rewritten | `cloud_train.py:2163` | Pointer → versioned dir → output_dir priority; optimizer from verified dir |
| `save_best_milestone()` COMMITTED marker | `cloud_train.py:1644` | Writes COMMITTED after atomic promotion |
| `save_checkpoint()` retention call | `cloud_train.py:1975` | Calls cleanup after pointer published |
| `TaraStreamingJsonlDataset.__init__()` buffer guard | `cloud_train.py:649` | Warning when buffer_size > 1 + active resume cursor |
| `TaraStreamingJsonlDataset.__iter__()` line-skip fix | `cloud_train.py:803` | Two-stage fast-forward: `<` pre-resume lines; skip resume line only when chunk_idx==0 |
| `tests/test_phase1_1.py` new | 1082 lines | 27 regression tests for all 6 gates |

---

## Unresolved Risks and Known Limitations

| Risk | Severity | Notes |
|------|----------|-------|
| `buffer_size > 1` + resume: non-deterministic replay window of up to `buffer_size` samples | LOW | Warning printed; production uses `buffer_size=1` (default). Samples replayed, not dropped. |
| GDrive FUSE `os.rename` fallback to `shutil.copytree` can be interrupted | LOW | Leaves staging dir with no COMMITTED marker — ignored on recovery. Prior versioned dir always intact. |
| Retention policy uses `os.listdir` — not atomic on FUSE | LOW | Race window is theoretical (single-process). Worst case: one extra old dir survives one cycle. |
| `active_chunk_idx` cursor is the last yielded chunk, not the last optimizer-committed chunk | ACCEPTED | At-least-once design: one window of samples may replay on resume. Documented semantics. |

---

## Final Gate Summary

| Gate | Topic | Result |
|------|-------|--------|
| G1 | Token / chunk completeness | **✅ PASS** |
| G2 | Ordering and cursor correctness | **✅ PASS** |
| G3 | Checkpoint durability | **✅ PASS** |
| G4 | Checkpoint / ledger consistency | **✅ PASS** |
| G5 | Best Model safety | **✅ PASS** |
| G6 | Storage retention | **✅ PASS** |

**Overall Phase 1.1 Acceptance: ✅ PASS**
**49/49 tests pass. All 6 blocking issues resolved with genuine passing tests.**

> Phase 1 remains on `phase1-data-safety-fixes`. Not merged to `main`.
> No production checkpoints, datasets, or Google Drive artifacts were modified.
> Production training may not begin until the pre-training gate (HF dataset verification, tokenizer audit) completes.
