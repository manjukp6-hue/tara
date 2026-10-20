# TARA Phase 3.1 — Final Acceptance and Colab Persistence Gate

**Branch:** `phase3-incremental-training`
**Starting commit:** `00b19b2` (clean working tree before Phase 3.1 began)
**Phase 3.1 commits:**
- `8b9524c` — Phase 3.1-A/B: Wire ShardLineIndex byte-seek into `__iter__`; add `DatasetSnapshot`; fix zero-offset guard and erroneous chunk>0 fallback seek
- `e54e518` — Phase 3.1 tests: 22/22 acceptance tests pass; subprocess resume helper for Gate C2

**Report generated:** 2026-10-20

---

## Pre-Phase 3.1 Baseline

Prior to any Phase 3.1 changes the suite was confirmed at:
```
122 passed, 7 subtests passed
```
(85 Phase 1/1.1/2/2.1 + 37 Phase 3 tests)

---

## Changed Files

### `cloud_train.py` — commit `8b9524c`

| Area | Change | Lines (approx) |
|------|--------|----------------|
| New class `DatasetSnapshot` | Persistent shard catalog with content SHA-256 tracking; load/save atomic; diff against live discovery | +144 lines after `ShardLineIndex` |
| `TaraStreamingJsonlDataset.__init__` | Added `dataset_snapshot: Optional[DatasetSnapshot] = None` parameter; instantiate `self._line_index = ShardLineIndex(dataset_dir) if dataset_dir else None` | ~L922 |
| `TaraStreamingJsonlDataset.__iter__` — shard SHA block | Added `self._snapshot.record_shard_hash(shard_name, shard_sha)` call when snapshot present | ~L1095 |
| `TaraStreamingJsonlDataset.__iter__` — byte-seek block | Replaced 4-line `target_resume_byte`-only seek with full `ShardLineIndex`-backed seek with index fallback; chunk==0 seeks to start of next line via index[line+1] or checkpoint byte; chunk>0 seeks to start of resume line via index[resume_line] | ~L1155–1215 |
| **Bug fix — zero-offset guard** | Changed `0 < offsets[target_resume_line]` → `0 <= offsets[target_resume_line]` so line 0 (byte offset = 0) is correctly indexed | L1178 |
| **Bug fix — erroneous chunk>0 fallback seek** | Removed the `elif target_resume_chunk > 0` branch in the no-index fallback that incorrectly used `target_resume_byte` (post-line offset) as a seek destination. For chunk>0 without an index, sequential scan is the correct and safe path | L1199–1202 removed |
| Sequential filter condition | Changed `if is_resume_target and target_resume_byte <= 0:` → `if is_resume_target and start_line_idx == 0:` to correctly bypass the line-skip filter when a byte-seek positioned the file pointer past or at the resume line | L1247 |
| `main()` — snapshot initialization | Instantiate `DatasetSnapshot(args.output_dir)`, call `initialize_from_discovery()`, log catalog diffs, pass `dataset_snapshot=dataset_snapshot` to dataset constructor | ~L3451–3475 |

### `tests/test_phase3_1.py` — commit `e54e518`

22 new tests across 4 gate classes + 1 regression class (623 lines).

### `tests/helpers/phase3_1_train_helper.py` — commit `e54e518`

Subprocess helper used by Gate C2. Simulates bounded two-process (Process A → commit → Process B → resume) training using synthetic JSONL fixtures (138 lines).

---

## Bug Report — Regressions Introduced and Fixed

### Root Cause

During Phase 3.1-A, the new `ShardLineIndex`-backed byte-seek block introduced a fallback path (no-index case, `target_resume_chunk > 0`) that incorrectly called `f.seek(target_resume_byte)`. The checkpoint stores `target_resume_byte = f.tell()` **after** reading the resume line, i.e. the first byte of the **next** line. For `chunk == 0` this is the correct seek destination (whole line committed → skip to next). For `chunk > 0` the resume line has uncommitted chunks that must be re-parsed — seeking past the line silently omits them.

Additionally the guard `0 < offsets[target_resume_line]` in the ShardLineIndex path incorrectly rejected line 0 (whose byte offset is legitimately 0).

### Tests That Regressed

```
FAILED tests/test_phase2.py::TestH3ByteSeekValidityGuard::test_byte_seek_skipped_when_resume_chunk_nonzero
FAILED tests/test_phase2_1.py::TestGate4CursorSemantics::test_h_g4_6_valid_byte_with_nonzero_chunk_skips_byte_seek
```

Both tests use `resume_line_offset=0`, `resume_chunk_idx > 0`, `resume_byte_offset = <post-line byte>` and no `dataset_dir` (no index file). The erroneous fallback `elif` block seeked to post-line byte → resume line was skipped → chunks from that line not yielded.

### Fix Applied

1. **Removed** the erroneous `elif target_resume_chunk > 0` fallback seek in the no-index path. For chunk>0 without an index, `sought` remains `False` and the safety fallback (`f.seek(0); start_line_idx = 0`) runs, allowing the sequential filter and per-chunk filter to navigate correctly.
2. **Changed** the ShardLineIndex guard from `0 < offsets[target_resume_line]` to `0 <= offsets[target_resume_line]` so line 0 is correctly handled when an index is present.

### Verification

After the fix:
```
2 previously failing tests: PASSED
122 total prior tests: PASSED (122/122, 7 subtests passed)
```

---

## Gate A — Wire the actual ShardLineIndex

### Tests

| Test | Status | Description |
|------|--------|-------------|
| `test_A1_resume_beginning` | **PASS** | Resuming from line 0 chunk 0 (whole line 0 committed) yields only lines 1+ |
| `test_A2_resume_middle_line_committed` | **PASS** | Resuming from a middle line (chunk 0) byte-seeks to the correct file position; no prior lines yielded |
| `test_A3_mid_line_chunk_resume` | **PASS** | Resuming with `chunk_idx > 0` within a line uses sequential scan for that line; uncommitted chunks yielded, committed chunks skipped |
| `test_A4_stale_index_fallback_sequential` | **PASS** | When the index SHA does not match the shard SHA (stale index), byte-seek is skipped and sequential scan is used; yields are correct |
| `test_A5_first_yielded_unit_is_next_uncommitted` | **PASS** | The first training unit yielded after resume has `(line_idx, chunk_idx) == (target_resume_line, target_resume_chunk)` |
| `test_A6_sequential_fallback_no_index` | **PASS** | When `dataset_dir` is absent (no `ShardLineIndex`), sequential scan navigates to the correct resume point without seek |

**Gate A: 6/6 PASS**

### Byte-seek resume guarantee

- `chunk == 0`: index[resume_line + 1] used (start of next line) if available; checkpoint byte offset as fallback. Correct because the whole resume line is committed.
- `chunk > 0`: index[resume_line] used (start of resume line itself); line is re-read from start so the per-chunk filter (`chunk_idx < target_resume_chunk → skip`) works correctly.
- Stale/missing index: sequential scan from file start with per-line and per-chunk filters. No committed unit is re-applied; no uncommitted unit is silently lost.
- Stale byte offset (shard content changed): `ShardLineIndex` detects SHA mismatch and rebuilds the index from the current file content. The old byte offsets are not used.

---

## Gate B — Dataset Manifest Semantics

### Design

The Hugging Face dataset repository does **not** publish a signed remote manifest file. `DatasetSnapshot` does **not** invent a remote format.

**Discovery source used:** `HfApi.list_repo_files()` (or a local file glob for offline/test environments). The API response is the only authoritative shard list.

**What `DatasetSnapshot` records:**
- `repo_id`, `prefix` (dataset sub-folder), `revision` metadata when available
- Ordered shard filenames as returned by discovery
- Runtime-computed SHA-256 per shard — populated at the moment each shard is opened locally, not derived from remote metadata

**Persistence:** `<output_dir>/dataset_snapshot.json` — atomic write (`tmp → os.replace`) to survive crashes.

**Change detection:**
- Added shards: present in live catalog, absent from snapshot
- Removed shards: present in snapshot, absent from live catalog
- Content-modified shards: same filename, live SHA ≠ snapshot SHA → completion state is not reused

### Tests

| Test | Status | Description |
|------|--------|-------------|
| `test_B1_persists_and_reloads` | **PASS** | Snapshot written and reloaded round-trips all fields correctly |
| `test_B2_diff_added_removed_common` | **PASS** | `diff_against_live()` correctly categorizes added, removed, common shards |
| `test_B3_content_modified_shard_detected` | **PASS** | After shard content changes, `is_shard_content_changed()` returns `True` for that shard |
| `test_B4_first_time_hash_not_a_change` | **PASS** | Recording a hash for the first time is not flagged as a content change (no prior hash) |
| `test_B4b_record_shard_hash_only_saves_on_change` | **PASS** | `record_shard_hash()` only writes to disk when the hash value changes; no spurious writes |
| `test_B5_atomic_write` | **PASS** | `save()` writes to a tmp file then renames; no partial JSON is observable at the snapshot path |

**Gate B: 6/6 PASS**

---

## Gate C — End-to-End Incremental Stage Acceptance

Bounded synthetic datasets used throughout. No production datasets or HF downloads invoked.

### Tests

| Test | Status | Description |
|------|--------|-------------|
| `test_C1_stage0_shards_skipped_in_stage1` | **PASS** | Stage 0 shards recorded as completed in `ShardLedger`; Stage 1 iterator skips them entirely; Stage 1 units come only from Stage 1 shards |
| `test_C2_subprocess_resume_exact_continuation` | **PASS** | Process A trains N units and commits a checkpoint; Process B (fresh subprocess) restores all state and resumes from exactly unit N+1; no unit is re-applied, no unit is skipped |
| `test_C3_unit_ids_no_repeat_no_gap` | **PASS** | Across Stage 0 → Stage 1 → Stage 2 (each interrupted mid-shard and resumed), training-unit ID audit confirms: zero duplicates, zero gaps, correct per-stage boundaries |

**Gate C: 3/3 PASS**

### Replay guarantee for uncommitted work

When a session ends before a checkpoint commit (e.g. Colab disconnect), the units processed since the last checkpoint are **not** durably committed. On the next session:
- The cursor is restored to the last committed position (`resume_line_offset`, `resume_chunk_idx` from checkpoint).
- Units between the last committed cursor and the session-end point are **replayed** — they receive another optimizer update.
- This is the explicit guarantee: committed units are never re-applied; uncommitted units since the last checkpoint boundary may be replayed at most once.
- The window of potential replay equals one `--save-steps` interval of training work.

---

## Gate D — Colab / Google Drive Persistence

### Honest Limitation Statement

The GDrive durability path (`--gdrive-mirror-dir` / `save_checkpoint` → `shutil.copy2` to the mirror path) was tested using local filesystem fixtures only. **No live Google Drive or Colab runtime was available for this Phase 3.1 acceptance run.**

The following is NOT claimed:
- That GDrive uploads actually executed during tests
- That Colab session resumption was live-tested
- That Drive API failures (quota, auth) are handled

The following IS implemented and tested locally:
- `save_checkpoint()` copies the checkpoint atomically to `--gdrive-mirror-dir` when the path is configured
- `test_D1_gdrive_mirror_attempted_when_configured`: verifies the mirror copy is made when configured
- `test_D2_gdrive_limitation_statement`: documents the scope limitation explicitly

**Gate D: 2/2 PASS (local filesystem only; GDrive live test not claimed)**

---

## Final Test Results

### Phase 3.1 tests (22 new)

```
Command: C:\Python314\python.exe -m pytest tests/test_phase3_1.py -v --tb=short
Result:  22 passed in 13.01s
```

All 22 passed:
- Gate A (ShardLineIndex wired): 6/6
- Gate B (DatasetSnapshot): 6/6
- Gate C (Incremental stages): 3/3
- Gate D (GDrive limitation): 2/2
- Regression checks: 5/5

### Full suite (144 tests = 122 prior + 22 new)

```
Command: C:\Python314\python.exe -m pytest tests/ -q --tb=short
Result:  144 passed, 7 subtests passed in 159.87s (0:02:39)
```

Zero failures. Zero regressions from Phase 1 through Phase 3.1.

---

## Remaining Risks

| Risk | Severity | Status |
|------|----------|--------|
| **GDrive live durability not tested** | Medium | Stated honestly — only local filesystem tested. Must be validated in an actual Colab session before production training. |
| **Remote manifest from HF not available** | Low | By design — `DatasetSnapshot` uses `HfApi.list_repo_files()` for discovery. No remote manifest format invented. This is the correct behavior per HF API design. |
| **Replay window for uncommitted work** | Low | One `--save-steps` interval of work may be replayed after a crash. This is a documented design property, not a bug. Reduce `--save-steps` to limit replay exposure. |
| **ShardLineIndex index coherence across shard mutations** | Low | Stale index (SHA mismatch) triggers a rebuild from current content. If shard content changes while a training session is active, the index is rebuilt on next open but the in-flight iterator is not affected. The next session would start clean. |
| **Windows `os.replace()` atomicity** | Very Low | On Windows, `os.replace()` is atomic within the same volume. Cross-volume checkpoint saves (rare) fall back to `shutil.copy2` + delete, which is not atomic. This edge case is handled in `save_checkpoint()` but was not specifically failure-tested. |

---

## Production Readiness Verdict

**NOT ready for production training merge.**

The Phase 3.1 acceptance gates have passed on local fixtures. Before merging to `main` or starting a full production training run, the following must be completed:

1. **Live Colab/GDrive test** — Run a bounded training session in Google Colab with `--gdrive-mirror-dir` pointing to an actual GDrive path; verify checkpoint is readable after session restart.
2. **Production dataset discovery validation** — Run `--dataset-prefix v1/` against the actual HF repo (without training) and verify `DatasetSnapshot` correctly captures the ordered shard list.
3. **Tokenizer and dataset content deduplication gate** — As required by the training pipeline governance policy, both the cross-version content deduplication check and the tokenizer collision audit must pass before any training run begins.
4. **Human review of Phase 3.1 commits** — `8b9524c` and `e54e518` must be reviewed before merge to `main`.

---

*Do not merge to `main`. Do not start production training. Review and live-test GDrive persistence first.*
