"""
TARA/LEARNING/self_upgrade.py

Permanent Self-Upgrade Architecture for TARA.
Enables TARA to iteratively improve its subsystems part-by-part:
- MODEL
- Tokenizer efficiency
- SKILLS
- KNOWLEDGE
- TOOLS
- MEMORY
- LEARNING
- Retrieval
- Inference efficiency
- Storage / indexing
- Performance

Workflow:
CURRENT TARA -> inspect -> identify limitation -> design in isolated workspace
-> implement -> test -> benchmark -> security check -> preserve rollback -> promote.
"""

import os
import sys
import json
import time
import shutil
import tempfile
import subprocess
import glob
from typing import Dict, List, Optional, Any, Callable

try:
    from TARA.ACCESS.operator.operator_profile import CANONICAL_CREATOR_ID
except Exception:
    CANONICAL_CREATOR_ID = "ROOT_OPERATOR"

from .security_guard import LearningSecurityGuard

class SelfUpgradeEngine:
    """
    Coordinates isolated testing, benchmarking, rollback backup,
    and safe integration of subsystem self-upgrades for TARA.
    """

    ALLOWED_SUBSYSTEMS = [
        "MODEL", "TOKENIZER", "SKILLS", "KNOWLEDGE",
        "TOOLS", "MEMORY", "LEARNING", "RETRIEVAL",
        "INFERENCE", "STORAGE", "PERFORMANCE"
    ]

    def __init__(self, repo_root: Optional[str] = None):
        if repo_root is None:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self.repo_root = repo_root
        self.rollback_dir = os.path.join(self.repo_root, "storage", "rollback")
        self.history_file = os.path.join(self.repo_root, "storage", "upgrade_history.json")
        os.makedirs(self.rollback_dir, exist_ok=True)
        self.security = LearningSecurityGuard()

    def inspect_subsystem(self, subsystem: str) -> Dict[str, Any]:
        """Inspects status, file counts, and metadata of a target subsystem."""
        sub = subsystem.upper()
        if sub not in self.ALLOWED_SUBSYSTEMS:
            raise ValueError(f"Unknown subsystem: {subsystem}. Allowed: {self.ALLOWED_SUBSYSTEMS}")

        path_map = {
            "MODEL": os.path.join(self.repo_root, "TARA", "MODEL"),
            "SKILLS": os.path.join(self.repo_root, "TARA", "SKILLS"),
            "KNOWLEDGE": os.path.join(self.repo_root, "TARA", "KNOWLEDGE"),
            "TOOLS": os.path.join(self.repo_root, "TARA", "TOOLS"),
            "MEMORY": os.path.join(self.repo_root, "TARA", "MEMORY"),
            "LEARNING": os.path.join(self.repo_root, "TARA", "LEARNING")
        }
        target_path = path_map.get(sub, self.repo_root)
        exists = os.path.exists(target_path)
        file_count = len(os.listdir(target_path)) if exists and os.path.isdir(target_path) else 0

        return {
            "subsystem": sub,
            "path": target_path,
            "exists": exists,
            "file_count": file_count,
            "permanent_identity": "TARA",
            "creator": CANONICAL_CREATOR_ID
        }

    def execute_self_upgrade(
        self,
        subsystem: str,
        upgrade_id: str,
        modified_files: Dict[str, str],
        test_scripts: Dict[str, str],
        benchmark_fn: Optional[Callable[[], bool]] = None,
        creator_id: str = CANONICAL_CREATOR_ID
    ) -> Dict[str, Any]:
        """
        Executes an upgrade candidate in an isolated temporary workspace.
        Requires:
        1. Rulebook / Creator protection verified.
        2. All test scripts pass in isolated environment.
        3. Benchmark verifies non-regression.
        4. Rollback copy preserved before committing changes.
        """
        sub = subsystem.upper()
        if sub not in self.ALLOWED_SUBSYSTEMS:
            raise ValueError(f"Invalid upgrade subsystem: {subsystem}")

        # 1. Security validation: Verify Rulebook invariant
        for rel_path in modified_files.keys():
            abs_target = os.path.join(self.repo_root, rel_path)
            self.security.assert_rulebook_protected(abs_target)

        # 2. Isolated Workspace Testing
        with tempfile.TemporaryDirectory(prefix="tara_upgrade_sandbox_") as tmp_dir:
            for rel_path, code_str in modified_files.items():
                dest = os.path.join(tmp_dir, rel_path)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                with open(dest, "w", encoding="utf-8") as f:
                    f.write(code_str)

            for t_name, t_code in test_scripts.items():
                t_dest = os.path.join(tmp_dir, t_name)
                os.makedirs(os.path.dirname(t_dest), exist_ok=True)
                with open(t_dest, "w", encoding="utf-8") as f:
                    f.write(t_code)

                proc = subprocess.run(
                    [sys.executable, t_dest],
                    capture_output=True,
                    text=True,
                    cwd=tmp_dir,
                    timeout=30
                )
                if proc.returncode != 0:
                    self.security.log_event("SELF_UPGRADE_FAILED", severity="ERROR", details={
                        "upgrade_id": upgrade_id,
                        "test": t_name,
                        "error": proc.stderr[:500]
                    })
                    return {
                        "status": "FAILED",
                        "reason": f"Test {t_name} failed in sandbox: {proc.stderr[:300]}",
                        "upgrade_id": upgrade_id
                    }

        # 3. Optional benchmark check
        if benchmark_fn:
            try:
                bm_pass = benchmark_fn()
                if not bm_pass:
                    return {
                        "status": "FAILED",
                        "reason": "Benchmark comparison indicated regression against CURRENT TARA.",
                        "upgrade_id": upgrade_id
                    }
            except Exception as e:
                return {
                    "status": "FAILED",
                    "reason": f"Benchmark evaluation error: {e}",
                    "upgrade_id": upgrade_id
                }

        # 4. Preserve Rollback Backup
        ts = int(time.time())
        upgrade_backup_dir = os.path.join(self.rollback_dir, f"{upgrade_id}_{ts}")
        os.makedirs(upgrade_backup_dir, exist_ok=True)

        for rel_path in modified_files.keys():
            actual_file = os.path.join(self.repo_root, rel_path)
            if os.path.exists(actual_file):
                backup_dest = os.path.join(upgrade_backup_dir, rel_path)
                os.makedirs(os.path.dirname(backup_dest), exist_ok=True)
                shutil.copy2(actual_file, backup_dest)

        # 5. Apply validated upgrade into CURRENT TARA
        for rel_path, code_str in modified_files.items():
            actual_file = os.path.join(self.repo_root, rel_path)
            os.makedirs(os.path.dirname(actual_file), exist_ok=True)
            with open(actual_file, "w", encoding="utf-8") as f:
                f.write(code_str)

        # 6. Record Upgrade History
        record = {
            "upgrade_id": upgrade_id,
            "subsystem": sub,
            "timestamp": time.time(),
            "iso_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "files_modified": list(modified_files.keys()),
            "rollback_path": os.path.relpath(upgrade_backup_dir, self.repo_root),
            "creator_id": creator_id,
            "status": "PROMOTED_TO_CURRENT_TARA"
        }
        history = []
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    history = json.load(f)
            except Exception:
                history = []
        history.append(record)
        with open(self.history_file, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        self.security.log_event("SELF_UPGRADE_PROMOTED", severity="INFO", details=record)

        return {
            "status": "SUCCESS",
            "upgrade_id": upgrade_id,
            "subsystem": sub,
            "permanent_identity": "TARA",
            "rollback_dir": upgrade_backup_dir
        }

    def rollback_upgrade(self, upgrade_id: str) -> bool:
        """Restores a previous state from rollback backup."""
        matching_dirs = sorted(glob.glob(os.path.join(self.rollback_dir, f"{upgrade_id}_*")))
        if not matching_dirs:
            return False

        latest_backup = matching_dirs[-1]
        for root, _, files in os.walk(latest_backup):
            for file in files:
                src_path = os.path.join(root, file)
                rel_path = os.path.relpath(src_path, latest_backup)
                dest_path = os.path.join(self.repo_root, rel_path)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.copy2(src_path, dest_path)

        return True
