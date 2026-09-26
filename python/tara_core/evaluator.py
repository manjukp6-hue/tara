"""
python/tara_core/evaluator.py

Evaluation, Verification, Recovery, and Uncertainty for TARA Core:
- Task Verification: checks whether execution results actually satisfy the requested objective.
- Recovery / Replanning: bounded, safe error correction (e.g. path normalization) without bypassing rules.
- Uncertainty Detection: explicitly states lack of verified evidence instead of hallucinating.
- Self-Evaluation: pre-finalization consistency review against user goal and tool evidence.
"""

import os
import re
from typing import Dict, Any, List, Optional, Tuple


class TaskVerifier:
    """Verifies that an execution result actually satisfies the requested objective."""

    @staticmethod
    def verify(
        action_intent: str,
        target_name: str,
        result: Any,
        expected_criteria: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validates output against objective criteria.
        Returns {"satisfied": bool, "verification_notes": str}.
        """
        if not isinstance(result, dict):
            return {"satisfied": True, "verification_notes": "Non-dictionary result accepted."}

        # Check explicit error or failure status
        if result.get("status") in ("ERROR", "FAILURE") or result.get("success") is False:
            return {
                "satisfied": False,
                "verification_notes": f"Execution reported failure: {result.get('error') or result.get('reason', 'Unknown error')}"
            }

        # Target-specific post-condition checks
        if target_name == "file_inspector":
            if not result.get("exists"):
                return {
                    "satisfied": False,
                    "verification_notes": f"File inspector did not locate target file: {result.get('file_path')}"
                }
            if result.get("size_bytes", 0) < 0 or result.get("line_count", 0) < 0:
                return {
                    "satisfied": False,
                    "verification_notes": "File metadata contains negative size or line counts."
                }
            return {
                "satisfied": True,
                "verification_notes": f"Verified file exists with {result.get('line_count')} lines ({result.get('size_bytes')} bytes)."
            }

        elif target_name == "hash_verifier":
            digest = result.get("hash")
            if not digest or not re.match(r"^[a-fA-F0-9]{32,64}$", str(digest)):
                return {
                    "satisfied": False,
                    "verification_notes": f"Hash verifier did not return a valid hexadecimal digest: {digest}"
                }
            if "match" in result and result.get("match") is False:
                return {
                    "satisfied": False,
                    "verification_notes": f"Hash mismatch: expected {result.get('expected_hash')} but got {result.get('computed_hash')}"
                }
            return {
                "satisfied": True,
                "verification_notes": f"Cryptographic digest verified ({len(digest) * 4}-bit): {digest[:16]}..."
            }

        elif target_name == "diagnostics":
            if not result.get("cpu_healthy") or result.get("memory_usage_mb", 0) <= 0:
                return {
                    "satisfied": False,
                    "verification_notes": "Telemetry reported unhealthy CPU or zero memory consumption."
                }
            return {
                "satisfied": True,
                "verification_notes": f"System healthy: CPU ok, RAM={result.get('memory_usage_mb')} MB, Temp={result.get('temperature_c')}C."
            }

        elif target_name == "developer":
            if result.get("valid") is False:
                return {
                    "satisfied": False,
                    "verification_notes": f"Code validation failed: {result.get('error')}"
                }
            return {
                "satisfied": True,
                "verification_notes": "Syntax validation passed successfully."
            }

        elif target_name == "device":
            if not result.get("safety_bounds_checked"):
                return {
                    "satisfied": False,
                    "verification_notes": "Device safety bounds check failed."
                }
            return {
                "satisfied": True,
                "verification_notes": f"G-code command verified within safety bounds for {result.get('kinematics')}."
            }

        return {
            "satisfied": True,
            "verification_notes": "Standard execution returned success."
        }


class RecoveryEngine:
    """Provides safe, bounded recovery and replanning for correctable execution failures."""

    def __init__(self, repo_root: str):
        self.repo_root = repo_root

    def attempt_recovery(
        self,
        tool_name: str,
        params: Dict[str, Any],
        error_msg: str
    ) -> Tuple[bool, Dict[str, Any], str]:
        """
        Attempts safe parameter healing for correctable errors.
        Returns (can_retry, healed_params, recovery_strategy).
        """
        # Case 1: File not found due to relative vs repo root path
        if tool_name in ("file_inspector", "hash_verifier") and "does not exist" in error_msg.lower():
            raw_path = params.get("file_path", "")
            # Try prepending or stripping repo root
            candidate_1 = os.path.join(self.repo_root, raw_path)
            candidate_2 = raw_path.strip("/\\")

            if os.path.exists(candidate_1):
                healed = dict(params)
                healed["file_path"] = candidate_1
                return True, healed, f"Normalized relative file path to repository root: {candidate_1}"

            norm_cand_2 = os.path.join(self.repo_root, candidate_2)
            if os.path.exists(norm_cand_2):
                healed = dict(params)
                healed["file_path"] = norm_cand_2
                return True, healed, f"Trimmed path separators and resolved against repo root: {norm_cand_2}"

        # Case 2: Code snippet surrounded by markdown backticks in developer skill
        if tool_name == "developer" and "syntax" in error_msg.lower():
            code = params.get("code", "")
            cleaned = re.sub(r"^```(?:python)?\s*", "", code.strip(), flags=re.MULTILINE)
            cleaned = re.sub(r"\s*```$", "", cleaned.strip(), flags=re.MULTILINE)
            if cleaned != code:
                healed = dict(params)
                healed["code"] = cleaned
                return True, healed, "Stripped markdown markdown fences from Python code"

        return False, params, "No safe autonomous recovery strategy available."


class UncertaintyDetector:
    """Detects lack of verified evidence and formulates calibrated uncertainty notices."""

    @staticmethod
    def evaluate_uncertainty(
        query: str,
        retrieved_knowledge: List[Dict[str, Any]],
        retrieved_memory: List[Dict[str, Any]]
    ) -> Optional[str]:
        """
        Returns an uncertainty statement if the system lacks knowledge on an explicit factual query.
        """
        q_lower = query.lower()
        factual_indicators = [
            r"\b(who|when|where|what is the capital of|how many|which year|history of|prime minister|minister of)\b",
            r"\b(specifications? of|creator of|born in|invented|president of)\b"
        ]
        is_factual_query = any(re.search(p, q_lower) for p in factual_indicators)

        # Check if retrieved knowledge actually contains relevant query terms
        meaningful_terms = [w for w in re.findall(r'\b\w+\b', q_lower) if len(w) >= 4 and w not in ["what", "when", "where", "which", "about", "tell", "explain", "minister"]]
        has_relevant = False
        if retrieved_knowledge and meaningful_terms:
            for k in retrieved_knowledge:
                c = str(k.get("content", "")).lower() + " " + str(k.get("subject", "")).lower()
                if any(t in c for t in meaningful_terms):
                    has_relevant = True
                    break
        elif retrieved_knowledge and not meaningful_terms:
            has_relevant = True

        if is_factual_query and (not retrieved_knowledge or not has_relevant):
            return (
                f"I do not have verified knowledge in my current database regarding '{query.strip()}'. "
                "To maintain strict epistemic integrity, I will not speculate or generate unverified claims."
            )

        return None


class SelfEvaluator:
    """Pre-finalization check ensuring the response addresses user intent without contradictions."""

    @staticmethod
    def evaluate_and_refine(
        user_input: str,
        tool_or_skill: Optional[Dict[str, Any]],
        result: Any,
        candidate_response: str,
        verification_report: Dict[str, Any]
    ) -> str:
        """
        Evaluates candidate response consistency against tool evidence.
        Revises response if a factual contradiction is detected.
        """
        # If verification failed but candidate response claimed success:
        if not verification_report.get("satisfied", True):
            note = verification_report.get("verification_notes", "Validation check failed")
            if not candidate_response.startswith("[ERROR") and not candidate_response.startswith("[BLOCKED"):
                return f"[TASK_VERIFICATION_FAILURE]: {note} (Response revised to prevent ungrounded success claim)"

        # Ensure tool results match output statements
        if isinstance(result, dict) and tool_or_skill and tool_or_skill.get("name") == "file_inspector":
            if result.get("exists") and "does not exist" in candidate_response.lower():
                return f"File '{result.get('file_path')}' verified exists ({result.get('line_count')} lines)."

        return candidate_response
