# TARA Phase 3.2 — Final Production Acceptance Report

**Branch:** `phase3-incremental-training`  
**Starting Commit:** `3561084`  
**Phase 3.2 Commit:** `1d9fb86`  
**Date:** 2026-10-20  

---

## 1. Executive Summary & Verification of Baseline

Prior to conducting the Phase 3.2 audit, branch HEAD and historical commits were verified:
- `8b9524c`: *Phase 3.1-A/B: Wire ShardLineIndex byte-seek into `__iter__`; add DatasetSnapshot for content-tracked shard catalog; fix zero-offset guard (0 <= for line 0) and remove erroneous chunk>0 fallback seek that used post-line byte offset*
- `e54e518`: *Phase 3.1 tests: 22/22 acceptance tests pass; subprocess resume helper for Gate C2 end-to-end resume verification*
- `3561084`: *Add TARA_PHASE3_1_ACCEPTANCE_REPORT.md*

### Working-Tree Status
Clean working tree verified before and after test additions.

### Test Execution Summary
- **Phase 1, 1.1, 2, 2.1, 3, 3.1 Suite:** 144/144 tests passed (7 subtests passed).
- **Phase 3.2 Gate 3 Tests (`tests/test_phase3_2.py`):** 12/12 tests passed.
- **Combined Test Suite Total:** **156 passed**, 0 failed, 0 warnings.

---

## 2. Gate-by-Gate Production Acceptance Audit

### Gate 1 — Real Colab / Google Drive Persistence
- **Status:** **BLOCKED**
- **Evidence & Operational Boundary:**
  - Audit environment: Windows 11 host (local pairing environment).
  - Genuine Google Drive FUSE mount (`/content/drive`) and Google Colab kernel preemption are physically unavailable on local developer hardware.
  - In strict compliance with directive (*"If genuine Colab/Drive access is unavailable, mark this gate BLOCKED and provide an executable notebook test. Do not substitute a local filesystem test and call it a live Drive PASS"*), this gate is marked **BLOCKED**.
- **Deliverable:**
  - An executable, self-contained Colab acceptance notebook has been created: [`tests/colab_gdrive_acceptance_test.ipynb`](file:///C:/Users/Nawaz/Desktop/taracore/tests/colab_gdrive_acceptance_test.ipynb).
  - Notebook workflow:
    1. Mounts Google Drive (`drive.mount('/content/drive')`).
    2. Runs Process A: trains 5 steps, records data cursor, commits checkpoint and ledger into `/content/drive/MyDrive/tara_colab_persistence_test/stage_0`.
    3. Simulates preemption/crash via `os._exit(0)`.
    4. Runs Process B in fresh runtime: discovers checkpoint on Drive, restores model, optimizer, scaler, RNG, step, and cursor.
    5. Validates that previously committed training units are not re-applied and that resume starts at exact next uncommitted chunk.

---

### Gate 2 — Production Dataset Discovery Dry-Run
- **Status:** **BLOCKED / FAILED (Remote Access Blocked)**
- **Evidence & Execution Proof:**
  - Command executed: `C:\Python314\python.exe tests/scratch_gate2_discovery.py`
  - Target configured repository: `manjukp6/tara` (type: `dataset`)
  - Configured prefix: `v2/`
  - Authentication token: `HF_TOKEN` present in runtime environment.
  - Live API Response:
    ```
    [Discovery] Repository: manjukp6/tara
    [Discovery] Prefix:     v2/
    [Discovery] Token:      SET
    [Discovery] Calling HfApi.list_repo_files(repo_id='manjukp6/tara', repo_type='dataset')...
    [Discovery] ERROR: 401 Client Error.
    Repository Not Found for url: https://huggingface.co/api/datasets/manjukp6/tara/tree/main?recursive=true&expand=false
    huggingface_hub.errors.RepositoryNotFoundError: 401 Client Error.
    Invalid username or password / Repository Not Found.
    ```
- **Analysis:**
  - Discovered shard count: **0 shards** (inaccessible).
  - The repository `manjukp6/tara` is either not yet initialized as a dataset repo on Hugging Face, or the configured token lacks read permissions for that repository.
  - **Fail-Closed Safety Guard:** In `cloud_train.py` (lines 3435–3439), when remote discovery returns 0 shards and no local `.jsonl` files exist in `dataset_dir`, the training engine raises:
    `RuntimeError: [Dataset Discovery Fatal] Zero dataset shards discovered on Hugging Face repository 'manjukp6/tara' ... Cannot proceed with zero training shards.`
  - The engine correctly refused to train an empty or partial dataset.
  - Zero downloads, zero training, zero state corruption occurred.

---

### Gate 3 — Content Deduplication and Tokenizer Identity
- **Status:** **PASS**
- **Test Suite:** [`tests/test_phase3_2.py`](file:///C:/Users/Nawaz/Desktop/taracore/tests/test_phase3_2.py) (12/12 passed)
- **Verified Invariants:**
  1. `test_G3_1_identical_content_different_filenames_same_sha`: Identical byte content across distinct shard filenames produces identical SHA-256 digests.
  2. `test_G3_2_different_content_different_sha`: Different content produces distinct digests.
  3. `test_G3_3_dataset_snapshot_detects_content_modified_shard`: `DatasetSnapshot.is_shard_content_changed()` flags modified shards so completed-shard status is never erroneously inherited.
  4. `test_G3_9_duplicate_records_in_distinct_shards_yield_distinct_unit_ids`: Duplicate records across distinct shards yield distinct training unit IDs `(shard, line_idx, chunk_idx)`; duplicate handling is not done via silent omission.
  5. `test_G3_10_sha256_is_deterministic`: Digests are repeatable and canonical 64-character hex strings.
  6. `test_G3_4_save_checkpoint_copies_tokenizer_to_staging`: `save_checkpoint` copies `tokenizer.json` into the versioned staging directory.
  7. `test_G3_5_tokenizer_sha_matches_source`: Tokenizer copy is byte-identical to source file.
  8. `test_G3_6_model_weights_sha256_stored_in_checkpoint_state`: `checkpoint_state.json` contains authoritative `model_weights_sha256`.
  9. `test_G3_7_tokenizer_determinism_same_input_same_token_ids`: Tokenizer determinism guaranteed across identical inputs.
  10. `test_G3_8_two_different_vocab_tokenizers_produce_different_ids`: Mismatched tokenizers produce distinct token IDs, proving the necessity of checkpoint-level tokenizer validation.
  11. `test_G3_2_semantically_similar_records_are_not_auto_deduplicated`: Semantic variations are never silently merged or dropped.
  12. `test_G3_2b_exact_duplicate_records_in_same_shard_both_trained`: Duplicate lines inside a single shard are both yielded at their respective line indices.

---

### Gate 4 — Human-Readable Diff Review

A thorough code audit of commits `8b9524c` and `e54e518` was conducted. Seven findings were identified:

#### Finding 1: Broken Snapshot Diff Logging in `main()`
- **Severity:** High
- **Source Location:** [`cloud_train.py:3458-3471`](file:///C:/Users/Nawaz/Desktop/taracore/cloud_train.py#L3458-L3471)
- **Failure Scenario:**
  ```python
  dataset_snapshot = DatasetSnapshot(args.output_dir)
  dataset_snapshot.initialize_from_discovery(
      repo_id=dataset_repo,
      prefix=ds_prefix,
      ordered_shards=list(selected_shards),
  )
  if dataset_snapshot.ordered_shards != list(selected_shards):
      diff = dataset_snapshot.diff_against_live(list(selected_shards))
  ```
  `initialize_from_discovery` assigns `self.ordered_shards = list(ordered_shards)`. Therefore, the subsequent check `if dataset_snapshot.ordered_shards != list(selected_shards)` is unconditionally `False`. Added or removed shards between training sessions are never detected or logged in `main()`.
- **Recommended Fix:** Compute `diff = dataset_snapshot.diff_against_live(list(selected_shards))` *before* invoking `initialize_from_discovery(...)`.

---

#### Finding 2: Unmonitored Staging Directory Accumulation on Abrupt Process Termination
- **Severity:** Medium
- **Source Location:** [`cloud_train.py:2247-2252`](file:///C:/Users/Nawaz/Desktop/taracore/cloud_train.py#L2247-L2252), [`cloud_train.py:2627`](file:///C:/Users/Nawaz/Desktop/taracore/cloud_train.py#L2627)
- **Failure Scenario:**
  Staging directories are named `.ckpt_staging_{basename}_step{N}_{ts}_{pid}`.
  In `_cleanup_old_versioned_checkpoints`, the directory scanner explicitly filters:
  `if entry.startswith(prefix) and not entry.startswith("."):`
  If a Colab runtime terminates mid-staging (e.g., OOM during `model.safetensors` write), the dot-prefixed staging directory is never inspected or purged by `_cleanup_old_versioned_checkpoints`. On Colab Free instances with limited disk, multiple orphaned ~1.5 GB staging dirs could exhaust disk space.
- **Recommended Fix:** Add a garbage-collection pass in `_cleanup_old_versioned_checkpoints` to scan for `.ckpt_staging_*` directories belonging to dead PIDs or with timestamps older than a safe threshold.

---

#### Finding 3: Resumption at Shard Boundary Causes Unnecessary Shard Re-Download
- **Severity:** Medium
- **Source Location:** [`cloud_train.py:1089-1102`](file:///C:/Users/Nawaz/Desktop/taracore/cloud_train.py#L1089-L1102), [`cloud_train.py:3813-3829`](file:///C:/Users/Nawaz/Desktop/taracore/cloud_train.py#L3813-L3829)
- **Failure Scenario:**
  When the last training unit of shard $S_k$ is committed into an optimizer step, `current_cursor["active_shard"]` is recorded as $S_k$. Shard $S_k$ is then marked completed in the ledger and its local file is evicted to reclaim disk.
  If the process restarts, `resume_shard` is $S_k$.
  In `__iter__`:
  `is_resume_target = (shard_name == target_resume_shard)` (which evaluates to `True`).
  Because `is_resume_target` is `True`, line 1091 (`if not is_resume_target: continue`) does *not* skip the completed shard!
  Because the local file was evicted, line 1096 calls `download_single_shard_jit(...)`, re-downloading the entire shard from Hugging Face simply to seek to EOF and yield zero items.
- **Recommended Fix:** When a shard completes and all units are committed, advance `current_cursor["active_shard"]` to the subsequent shard in the catalog, or update the ledger skip condition so an evicted, fully-completed resume target shard advances immediately without re-downloading.

---

#### Finding 4: Incomplete Shard Index Fallback Scans Full File on End-of-Shard Boundary
- **Severity:** Low / Latency
- **Source Location:** [`cloud_train.py:1167-1175`](file:///C:/Users/Nawaz/Desktop/taracore/cloud_train.py#L1167-L1175)
- **Failure Scenario:**
  When `target_resume_chunk == 0`, the code looks for `next_line = target_resume_line + 1` in `offsets`.
  If `target_resume_line` was the last line in the shard, `next_line` is not in `offsets`.
  If `target_resume_byte` is also unrecorded or 0, `sought` remains `False`.
  The iterator falls back to `f.seek(0)` and scans all lines in the file sequentially only to discard them at `if line_idx < target_resume_line: continue`.
- **Recommended Fix:** If `next_line` exceeds `max(offsets.keys())`, seek directly to EOF (`f.seek(0, 2)`) and set `start_line_idx = next_line`, avoiding redundant file read.

---

#### Finding 5: Missing Tokenizer Identity Fingerprint in `checkpoint_state.json`
- **Severity:** Medium
- **Source Location:** [`cloud_train.py:2306-2330`](file:///C:/Users/Nawaz/Desktop/taracore/cloud_train.py#L2306-L2330), [`cloud_train.py:2764-2780`](file:///C:/Users/Nawaz/Desktop/taracore/cloud_train.py#L2764-L2780)
- **Failure Scenario:**
  While `tokenizer.json` is copied into the checkpoint directory, `checkpoint_state.json` does not record `tokenizer_sha256` or vocabulary size.
  During `try_resume_from_checkpoint`, the runtime validates model weights and optimizer step synchrony, but does not verify that the tokenizer in use matches the checkpoint's tokenizer.
  A user resuming with an altered tokenizer vocabulary would experience silent token drift.
- **Recommended Fix:** Compute and persist `tokenizer_sha256` in `checkpoint_state.json` and verify it during `try_resume_from_checkpoint`.

---

#### Finding 6: Redundant Full-Shard SHA-256 Recalculation on Re-Open
- **Severity:** Low / Performance
- **Source Location:** [`cloud_train.py:1105`](file:///C:/Users/Nawaz/Desktop/taracore/cloud_train.py#L1105)
- **Failure Scenario:**
  `compute_file_sha256` is called on the active shard whenever `__iter__` reaches it. For a 1 GiB shard, this incurs ~1–2 seconds of CPU I/O overhead.
  While this guarantees cryptographic safety, when reading multi-gigabyte shards across multiple epochs, repeatedly hashing unchanged shards introduces unnecessary I/O overhead.
- **Recommended Fix:** Cache `(mtime, size, sha256)` in `DatasetSnapshot` so repeated passes over immutable shards avoid redundant byte streaming.

---

#### Finding 7: Sequential Filter Redundant JSON Parsing on Line Skip
- **Severity:** Low
- **Source Location:** [`cloud_train.py:1257-1262`](file:///C:/Users/Nawaz/Desktop/taracore/cloud_train.py#L1257-L1262)
- **Failure Scenario:**
  When `target_resume_chunk > 0` and `target_resume_line` was fully consumed before crash, the line is read, decoded from JSON, tokenized, and only discarded in the inner token chunk loop.
- **Recommended Fix:** When `target_resume_chunk` indicates the entire line has been consumed, advance `target_resume_line` to `target_resume_line + 1` with `target_resume_chunk = 0` at checkpoint save time.

---

## 3. Production Readiness Verdict

```
================================================================================
FINAL PRODUCTION READINESS VERDICT: NOT READY
================================================================================
Deployment to production training remains UNAUTHORIZED.

Blocking Items:
1. Gate 1 (Colab/Google Drive live durability) is BLOCKED: requires execution
   of tests/colab_gdrive_acceptance_test.ipynb in a live Google Colab session.
2. Gate 2 (Production dataset discovery dry-run) is BLOCKED: remote repository
   'manjukp6/tara' returned HTTP 401 Unauthorized / Repository Not Found.
   Dataset repository access and credentials must be established.
3. Gate 4 findings:
   - Finding 1 (broken snapshot diff logging in main()) must be resolved.
   - Finding 3 (redundant re-download on shard boundary resume) must be resolved.
   - Finding 5 (tokenizer_sha256 verification in checkpoint_state.json) must be resolved.

Do not merge branch 'phase3-incremental-training' to 'main'.
================================================================================
```
