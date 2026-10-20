# TARA Phase 3.4.1 — Report: Checkpoint Selection Isolation & Multi-Stage Scope Verification

**Date:** 2026-10-20  
**Repository Branch:** `phase3-incremental-training`  
**Base Commit:** `001db4e2baa2b11fa859b718894a87b24d721986` (`001db4e`)  
**Remediation Commit:** `3ea978a895d52e6cfe2ca331d009b3ba05d46bbc` (`3ea978a`)  
**Working Tree Status:** Clean  
**Final Production Readiness Verdict:** **NOT READY (DEPLOYMENT UNAUTHORIZED)**

---

## 1. Problem Identification & Root Cause Analysis

### 1.1 The Issue
During initial Colab live persistence execution, `try_resume_from_checkpoint()` was invoked with:
`/content/drive/MyDrive/tara_colab_persistence_test/stage_0`
However, the discovery logs reported selecting:
`/content/drive/MyDrive/tara_colab_persistence_test/stage_interrupted_test.step5_1792498153`

### 1.2 Root Cause
1. **Unscoped Shared Pointer:** In `save_checkpoint()`, `latest_checkpoint.json` was written to `parent_dir` (`/content/drive/MyDrive/tara_colab_persistence_test`). When multiple stages or sibling experiments (`stage_0` and `stage_interrupted_test`) shared that common parent directory, whichever stage checkpointed last unconditionally overwritten `parent_dir/latest_checkpoint.json`.
2. **Missing Scope Verification:** In `try_resume_from_checkpoint()`, the discovery logic read `parent_dir/latest_checkpoint.json` and blindly accepted its target versioned directory without verifying that the pointer belonged to the requested `output_dir` basename or `stage_id`.
3. **Missing Candidate Validation:** The candidate evaluation loop in `try_resume_from_checkpoint()` did not validate `expected_stage_id` against `s_data["stage_id"]`, allowing cross-stage adoption when step counts and file structures aligned.

---

## 2. Technical Remediation

### 2.1 Directory-Scoped Pointer Publication (`cloud_train.py:2583-2615`)
- `save_checkpoint()` now writes `latest_checkpoint.json` directly inside `output_dir` (`checkpoint_dir/latest_checkpoint.json`) and records `stage_id`, `output_dir`, and `step`.
- For backward compatibility with legacy single-stage callers, it mirrors a copy to `parent_dir/latest_checkpoint.json`.

### 2.2 Scoped Pointer Resolution & Strict Candidate Validation (`cloud_train.py:2935-3120`)
- `try_resume_from_checkpoint()` now checks directory-scoped pointer (`checkpoint_dir/latest_checkpoint.json`) first before inspecting parent pointers.
- Added strict verification of pointer contents:
  1. Rejects pointers where `ptr_output` basename does not match target `output_dir_basename`.
  2. Rejects pointers where `ptr_stage` contradicts `expected_stage_id`.
  3. Rejects pointers whose `versioned_dir` name does not start with `{output_dir_basename}.step`.
- In the candidate evaluation loop:
  1. Validates that candidate directory matches target output scope (`checkpoint_dir`, `{output_dir_basename}.step*`, or GDrive mirror).
  2. Enforces `expected_stage_id`: if `expected_stage_id` is supplied, candidate checkpoints with differing `stage_id` are rejected immediately.
- Wired `expected_stage_id=args.stage_id` in `main()` (`cloud_train.py:3966`).

### 2.3 Retention Policy Isolation (`cloud_train.py:2780-2805`)
- `_cleanup_old_versioned_checkpoints()` now scopes pointer resolution to `output_dir_basename`, guaranteeing that cleaning up old versioned directories in `stage_0` will never read or delete pointers belonging to sibling stages.

---

## 3. Verification & Test Proofs

### 3.1 Dedicated Isolation Regression Suite (`tests/test_phase3_4_1.py`)
Created 5 new regression tests covering the required scenarios:
- `test_resume_stage_0_selects_exact_stage_0_state`: Resumes `stage_0` (step 5) and asserts exact model weights, optimizer state, and cursor matching `model_a`, rejecting sibling `stage_interrupted_test`.
- `test_resume_stage_interrupted_selects_exact_stage_interrupted_state`: Resumes `stage_interrupted_test` (step 9) and asserts exact step 9, distinct model weights, and distinct cursor.
- `test_sibling_pointer_in_parent_rejected_for_stage_0`: Injects a cross-directory parent pointer pointing to `stage_interrupted_test`; asserts `stage_0` rejects it and cleanly recovers its own step 5 checkpoint.
- `test_stale_corrupt_pointer_falls_back_to_valid_versioned_dir`: Verifies corrupted pointer fails safe without adopting unrelated sibling checkpoint.
- `test_stage_id_mismatch_fails_closed`: Verifies that querying with an mismatched `expected_stage_id` fails closed (`was_resumed=False`).

**Test Execution Result:**
- `5 passed in 17.82s` (0 warnings).

### 3.2 Full Repository Regression
Executed full test suite across all 13 test modules:
```bash
C:\Python314\python.exe -m pytest tests/ -q --tb=short
```
**Result:**
- **178 passed, 7 subtests passed, 0 failed, 0 warnings** in 182.13s.

---

## 4. Colab Acceptance Notebook Hardening (`tests/colab_gdrive_acceptance_test.ipynb`)

1. **Drive-Backed Fixtures (Requirement 9):**
   - In Part 1, synthetic JSONL shards and tokenizer files are written directly to Google Drive (`tara_colab_persistence_test/fixtures/`).
   - Part 2 in the fresh runtime loads fixtures directly from Google Drive, proving complete recovery with zero reliance on ephemeral `/content` from the previous session.
2. **Two Intentionally Different Checkpoints (Requirement 4):**
   - Part 1 commits `stage_0` at step 5 and `stage_interrupted_test` at step 9 to Drive.
3. **Exact Stage Verification (Requirements 5 & 6):**
   - Part 2 verifies `stage_0` restores step 5 and cursor offset 5.
   - Part 2 verifies `stage_interrupted_test` restores step 9 independently.
4. **Real Streaming Dataset Iterator (Requirement 8):**
   - Instead of manually slicing strings, Part 2 instantiates `TaraStreamingJsonlDataset` with `resume_line_offset=5` and iterates through the remaining units (`line_5_chunk_0`, `line_6_chunk_0`).
   - Proves zero duplication of lines 0..4 and complete training of uncommitted units to step 7.

---

## 5. Summary & Remaining Blockers

| Gate | Status | Detail |
| :--- | :---: | :--- |
| **Finding: Checkpoint Selection Isolation** | **RESOLVED & VERIFIED (PASS)** | Directory-scoped pointer + strict scope & `stage_id` validation. |
| **Gate 1: Hugging Face Discovery** | **BLOCKED** | Awaiting Colab output from `tara_sync` secret discovery run. |
| **Gate 2: Real Colab / GDrive Persistence** | **BLOCKED** | Ready for live Colab execution using hardened `colab_gdrive_acceptance_test.ipynb`. |

**Verdict:** **NOT READY (DEPLOYMENT UNAUTHORIZED)**
