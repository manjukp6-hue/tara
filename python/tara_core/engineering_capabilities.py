"""
python/tara_core/engineering_capabilities.py

Software Engineering, Data Engineering & Experiment Design Capabilities for TARA Core.
Provides production-grade implementations for:
1. Software Engineering (READ -> UNDERSTAND -> MODIFY -> TEST -> DEBUG -> VERIFY -> VERSION -> ROLLBACK)
2. Data Engineering (CSV/JSON/Databases, validation, transformation, normalization, indexing, querying)
3. Experiment Design (HYPOTHESIS -> EXPERIMENT -> OBSERVATION -> RESULT -> ANALYSIS -> CONCLUSION)
"""

import os
import sys
import ast
import csv
import json
import time
import math
import uuid
import logging
import threading
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger("TARA.EngineeringCapabilities")


# ----------------------------------------------------------------------------
# 20. Software Engineering Capability
# ----------------------------------------------------------------------------

@dataclass
class CodeAnalysis:
    is_valid_syntax: bool
    classes_defined: List[str]
    functions_defined: List[str]
    imported_modules: List[str]
    lines_of_code: int
    error_message: Optional[str] = None


class SoftwareEngineeringEngine:
    """Automates code understanding, AST parsing, sandboxed testing, and versioned modifications."""

    def __init__(self, backup_dir: Optional[str] = None):
        self.backup_dir = backup_dir or os.path.abspath(os.path.join(os.path.dirname(__file__), "../../storage/backups/code"))
        os.makedirs(self.backup_dir, exist_ok=True)
        self._snapshots: Dict[str, List[str]] = {} # file_path -> list of backup paths
        self._lock = threading.RLock()

    def understand_code(self, source_code: str) -> CodeAnalysis:
        """Parses Python AST to extract structural insights without side-effects."""
        try:
            tree = ast.parse(source_code)
            classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for n in node.names:
                        imports.append(n.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)

            lines = len([line for line in source_code.splitlines() if line.strip()])
            return CodeAnalysis(
                is_valid_syntax=True,
                classes_defined=classes,
                functions_defined=functions,
                imported_modules=sorted(list(set(imports))),
                lines_of_code=lines
            )
        except SyntaxError as se:
            return CodeAnalysis(
                is_valid_syntax=False,
                classes_defined=[],
                functions_defined=[],
                imported_modules=[],
                lines_of_code=0,
                error_message=f"SyntaxError at line {se.lineno}: {se.msg}"
            )

    def test_code_snippet(self, code: str, expected_variable: str, expected_value: Any) -> Dict[str, Any]:
        """Executes code snippet in a restricted namespace and verifies output state."""
        syntax_check = self.understand_code(code)
        if not syntax_check.is_valid_syntax:
            return {"status": "FAILED", "stage": "SYNTAX_CHECK", "error": syntax_check.error_message}

        restricted_globals: Dict[str, Any] = {"__builtins__": {
            "abs": abs, "len": len, "range": range, "str": str, "int": int, "float": float,
            "list": list, "dict": dict, "set": set, "sum": sum, "min": min, "max": max,
            "enumerate": enumerate, "zip": zip, "print": lambda *args: None
        }}
        local_scope: Dict[str, Any] = {}

        try:
            exec(code, restricted_globals, local_scope)
            actual_val = local_scope.get(expected_variable)
            passed = (actual_val == expected_value)
            return {
                "status": "SUCCESS" if passed else "FAILED",
                "stage": "VERIFICATION",
                "passed": passed,
                "expected": expected_value,
                "actual": actual_val
            }
        except Exception as ex:
            return {"status": "FAILED", "stage": "EXECUTION", "error": str(ex)}

    def create_snapshot_and_save(self, file_path: str, new_code: str) -> Dict[str, Any]:
        with self._lock:
            # 1. Snapshot existing if exists
            if os.path.exists(file_path):
                snap_id = f"snap_{int(time.time()*1000)}"
                snap_path = os.path.join(self.backup_dir, f"{os.path.basename(file_path)}.{snap_id}.bak")
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    old_data = f.read()
                with open(snap_path, "w", encoding="utf-8") as f:
                    f.write(old_data)
                if file_path not in self._snapshots:
                    self._snapshots[file_path] = []
                self._snapshots[file_path].append(snap_path)

            # 2. Write new
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_code)

            return {
                "status": "SAVED",
                "file_path": file_path,
                "snapshot_created": os.path.exists(file_path),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

    def rollback_last_snapshot(self, file_path: str) -> Dict[str, Any]:
        with self._lock:
            snaps = self._snapshots.get(file_path, [])
            if not snaps:
                return {"status": "FAILED", "error": f"No backup snapshots found for '{file_path}'"}
            last_snap = snaps.pop()
            if os.path.exists(last_snap):
                with open(last_snap, "r", encoding="utf-8") as f:
                    content = f.read()
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                return {"status": "ROLLBACK_SUCCESS", "restored_from": last_snap}
            return {"status": "FAILED", "error": "Snapshot file missing on disk"}

    # Backward/forward alias
    modify_file_with_safety_rollback = create_snapshot_and_save
    rollback_file = rollback_last_snapshot


# ----------------------------------------------------------------------------
# 21. Data Engineering Capability
# ----------------------------------------------------------------------------

class DataEngineeringEngine:
    """Ingests, validates, normalizes, transforms, and queries structured data sets."""

    @staticmethod
    def parse_and_validate_csv(csv_text: str, required_columns: List[str]) -> Dict[str, Any]:
        lines = csv_text.strip().splitlines()
        if not lines:
            return {"valid": False, "error": "CSV is empty"}
        reader = csv.DictReader(lines)
        fieldnames = reader.fieldnames or []
        missing = [col for col in required_columns if col not in fieldnames]
        if missing:
            return {"valid": False, "error": f"Missing required columns: {missing}", "found": fieldnames}

        rows = list(reader)
        return {
            "valid": True,
            "row_count": len(rows),
            "columns": fieldnames,
            "sample_rows": rows[:3]
        }

    @staticmethod
    def normalize_records(records: List[Dict[str, Any]], field_casts: Dict[str, type]) -> List[Dict[str, Any]]:
        """Cleans and casts dictionary records to standardized datatypes."""
        normalized = []
        for r in records:
            clean = dict(r)
            for f_name, f_type in field_casts.items():
                if f_name in clean:
                    try:
                        clean[f_name] = f_type(clean[f_name])
                    except (ValueError, TypeError):
                        clean[f_name] = None
            normalized.append(clean)
        return normalized

    @staticmethod
    def query_dataset(records: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Filters dataset records by key-value criteria."""
        results = []
        for r in records:
            match = True
            for k, expected_v in filters.items():
                if r.get(k) != expected_v:
                    match = False
                    break
            if match:
                results.append(r)
        return results


# ----------------------------------------------------------------------------
# 22. Experiment Design
# ----------------------------------------------------------------------------

@dataclass
class ExperimentResult:
    experiment_id: str
    hypothesis: str
    control_mean: float
    variant_mean: float
    uplift_percent: float
    statistically_significant: bool
    conclusion: str


class ExperimentDesignEngine:
    """Manages hypothesis-driven experiments, metric trials, and evidence-based conclusions."""

    def run_ab_experiment(
        self,
        experiment_id: str,
        hypothesis: str,
        control_samples: List[float],
        variant_samples: List[float],
        min_uplift_threshold: float = 0.05
    ) -> ExperimentResult:
        if not control_samples or not variant_samples:
            return ExperimentResult(
                experiment_id=experiment_id,
                hypothesis=hypothesis,
                control_mean=0.0,
                variant_mean=0.0,
                uplift_percent=0.0,
                statistically_significant=False,
                conclusion="Insufficient sample observations"
            )

        c_mean = sum(control_samples) / len(control_samples)
        v_mean = sum(variant_samples) / len(variant_samples)
        uplift = ((v_mean - c_mean) / max(1e-6, c_mean))

        # Basic variance check
        c_var = sum((x - c_mean)**2 for x in control_samples) / max(1, len(control_samples)-1)
        v_var = sum((x - v_mean)**2 for x in variant_samples) / max(1, len(variant_samples)-1)
        pooled_se = math.sqrt(max(1e-6, (c_var / len(control_samples)) + (v_var / len(variant_samples))))
        t_stat = (v_mean - c_mean) / max(1e-6, pooled_se)

        # Significant if |t| > 2.0 and uplift >= threshold
        is_sig = (abs(t_stat) > 2.0) and (uplift >= min_uplift_threshold)
        conclusion = (
            f"Hypothesis confirmed: Variant outperforms control by {uplift*100:.1f}% (t={t_stat:.2f})"
            if is_sig else
            f"Hypothesis not supported: Uplift {uplift*100:.1f}% below threshold or statistically insignificant"
        )

        return ExperimentResult(
            experiment_id=experiment_id,
            hypothesis=hypothesis,
            control_mean=round(c_mean, 4),
            variant_mean=round(v_mean, 4),
            uplift_percent=round(uplift * 100.0, 2),
            statistically_significant=is_sig,
            conclusion=conclusion
        )
