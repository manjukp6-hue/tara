"""
TARA/MEMORY/memory_engine.py

Canonical Episodic & Procedural Memory Engine for TARA Core (NVIDIA Voyager Architecture).
Maintains structured trajectory logs outside model weights under canonical TARA/MEMORY/:
- Intent & Action taken
- Tool/Skill inputs & outputs
- Success / Failure reflections
- Provenance & Actor verification
- Queryable historical memory & procedural recall
- Automatic migration from legacy storage/memory/
"""

import os
import json
import time
import shutil
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any

try:
    from TARA.ACCESS.operator.operator_profile import CANONICAL_CREATOR_ID
except Exception:
    CANONICAL_CREATOR_ID = "ROOT_OPERATOR"

class MemoryEngine:
    """Canonical Memory Engine for TARA, storing episodes and procedures in TARA/MEMORY/."""

    def __init__(self, memory_dir: Optional[str] = None, legacy_dir: Optional[str] = None):
        if memory_dir is None:
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            memory_dir = os.path.join(base_dir, "TARA", "MEMORY")
        
        if legacy_dir is None:
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            legacy_dir = os.path.join(base_dir, "storage", "memory")

        self.memory_dir = os.path.abspath(memory_dir)
        self.legacy_dir = os.path.abspath(legacy_dir)
        os.makedirs(self.memory_dir, exist_ok=True)

        self.episodes_file = os.path.join(self.memory_dir, "episodes.jsonl")
        self.procedures_file = os.path.join(self.memory_dir, "procedures.jsonl")

        # Automatically migrate legacy memory if canonical files do not yet exist
        self._auto_migrate_legacy_data()

    def _auto_migrate_legacy_data(self):
        """Migrates historical episodes/procedures from storage/memory/ if present."""
        if not os.path.exists(self.episodes_file) and os.path.exists(self.legacy_dir):
            legacy_episodes = os.path.join(self.legacy_dir, "episodes.jsonl")
            if os.path.exists(legacy_episodes) and os.path.getsize(legacy_episodes) > 0:
                try:
                    with open(legacy_episodes, "r", encoding="utf-8") as src, \
                         open(self.episodes_file, "w", encoding="utf-8") as dst:
                        for line in src:
                            if line.strip():
                                rec = json.loads(line.strip())
                                rec["migrated_from"] = "legacy_storage_memory"
                                dst.write(json.dumps(rec, ensure_ascii=False) + "\n")
                except Exception:
                    pass

        if not os.path.exists(self.procedures_file) and os.path.exists(self.legacy_dir):
            legacy_procedures = os.path.join(self.legacy_dir, "procedures.jsonl")
            if os.path.exists(legacy_procedures) and os.path.getsize(legacy_procedures) > 0:
                try:
                    shutil.copy2(legacy_procedures, self.procedures_file)
                except Exception:
                    pass

    def record_episode(
        self,
        actor_id: str,
        intent: str,
        action: str,
        parameters: Dict[str, Any],
        outcome: str,
        observations: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        reflection: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Records an execution episode trajectory.
        """
        episode_id = "ep_" + hashlib.sha256(f"{action}_{time.time()}".encode()).hexdigest()[:12]
        record = {
            "episode_id": episode_id,
            "actor_id": actor_id,
            "intent": intent,
            "action": action,
            "parameters": parameters,
            "outcome": outcome,  # SUCCESS / FAILURE
            "observations": observations or {},
            "error": str(error) if error else None,
            "reflection": reflection or ("Task succeeded" if outcome == "SUCCESS" else "Task failed"),
            "timestamp": datetime.now().isoformat()
        }

        with open(self.episodes_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return record

    def query_episodes(
        self,
        query: Optional[str] = None,
        actor_id: Optional[str] = None,
        outcome: Optional[str] = None,
        limit: int = 10,
        scope: str = "user"
    ) -> List[Dict[str, Any]]:
        """
        Queries historical episodes by keyword or outcome (most recent first).
        Enforces cross-user isolation:
        - When actor_id is provided for a standard user, strictly returns episodes belonging to that actor_id.
        - Normal users NEVER receive another user's episodes.
        - When actor_id == "ROOT_OPERATOR" (Creator) or scope == "system", allows access to creator / system-scoped episodes.
        - When actor_id is None, allows unconstrained administrative / legacy test queries.
        """
        if not os.path.exists(self.episodes_file):
            return []

        all_records = []
        with open(self.episodes_file, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    all_records.append(json.loads(line_str))
                except Exception:
                    continue

        matches = []
        q_lower = query.lower() if query else None
        q_words = [w.strip(".,!?\"'()[]{}") for w in q_lower.split() if len(w.strip(".,!?\"'()[]{}")) >= 3] if q_lower else []

        for rec in reversed(all_records):
            # Strict User Isolation
            rec_actor = rec.get("actor_id")
            if actor_id is not None:
                if actor_id == "ROOT_OPERATOR":
                    if scope == "user" and rec_actor not in ("ROOT_OPERATOR", "system", "TARA"):
                        continue
                else:
                    if rec_actor != actor_id:
                        continue

            if outcome and rec.get("outcome") != outcome:
                continue
            if q_lower:
                haystack = f"{rec.get('intent', '')} {rec.get('action', '')} {rec.get('parameters', '')} {rec.get('observations', '')}".lower()
                if q_lower in haystack or (q_words and any(w in haystack for w in q_words)):
                    matches.append(rec)
                else:
                    continue
            else:
                matches.append(rec)

            if len(matches) >= limit:
                break
        return matches

    def store_procedure(self, task_key: str, steps: List[Any], description: Optional[str] = None) -> Dict[str, Any]:
        """
        Stores proven multi-step procedural skills.
        """
        proc_id = "proc_" + hashlib.sha256(task_key.encode()).hexdigest()[:12]
        procedure = {
            "proc_id": proc_id,
            "task_key": task_key,
            "description": description or f"Procedure for {task_key}",
            "steps": steps,
            "updated_at": datetime.now().isoformat()
        }

        with open(self.procedures_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(procedure, ensure_ascii=False) + "\n")

        return procedure

    def get_procedure(self, task_key: str) -> Optional[Dict[str, Any]]:
        if not os.path.exists(self.procedures_file):
            return None
        with open(self.procedures_file, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    p = json.loads(line_str)
                    if p.get("task_key") == task_key:
                        return p
                except Exception:
                    continue
        return None

    def get_memory_stats(self) -> Dict[str, Any]:
        """Returns statistics on stored memory files."""
        ep_count = 0
        if os.path.exists(self.episodes_file):
            with open(self.episodes_file, "r", encoding="utf-8") as f:
                ep_count = sum(1 for line in f if line.strip())

        proc_count = 0
        if os.path.exists(self.procedures_file):
            with open(self.procedures_file, "r", encoding="utf-8") as f:
                proc_count = sum(1 for line in f if line.strip())

        return {
            "memory_dir": self.memory_dir,
            "episodes_count": ep_count,
            "procedures_count": proc_count,
            "episodes_file": self.episodes_file,
            "procedures_file": self.procedures_file
        }
