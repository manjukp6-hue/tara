"""
python/tara_core/nlu.py

Natural Language Understanding (NLU) & Slot Extraction for TARA Core:
- Upgrades lexical intent parsing so natural-language paraphrases map reliably to intents.
- Preserves deterministic safety invariants (CSAM, Content Safety, Creator Authority, Deletion, Mutation).
- Extracts structured entity slots: file paths, date/times, amounts, devices, hashes, target objects.
"""

import os
import re
from typing import Dict, Any, List, Optional, Tuple


class SlotExtractor:
    """Extracts structured entities/parameters from unstructured natural language."""

    @staticmethod
    def extract_file_path(text: str) -> Optional[str]:
        """Extracts file or directory paths from text."""
        # 1. Quoted paths: "path/to/file" or 'path/to/file'
        quoted = re.findall(r'["\']([^"\']+\.[a-zA-Z0-9_-]+|[^"\']*[/\\][^"\']*)["\']', text)
        if quoted:
            return quoted[0].strip()

        # 2. File with known extensions (.json, .py, .txt, .log, .md, .safetensors, .db, .csv, .yaml, .yml, .gcode)
        ext_match = re.search(r'([a-zA-Z0-9_\-./\\]+\.(?:json|py|txt|log|md|safetensors|db|csv|yaml|yml|gcode|xyz))\b', text, re.IGNORECASE)
        if ext_match:
            return ext_match.group(1).strip()

        # 3. Path with directory separators (e.g., storage/models/tara or storage\models\tara)
        path_match = re.search(r'([a-zA-Z0-9_\-]+[/\\][a-zA-Z0-9_\-./\\]+)', text)
        if path_match:
            return path_match.group(1).strip()

        return None

    @staticmethod
    def extract_hash(text: str) -> Optional[str]:
        """Extracts cryptographic hex digest (MD5=32, SHA1=40, SHA256=64)."""
        match = re.search(r'\b([a-fA-F0-9]{64}|[a-fA-F0-9]{40}|[a-fA-F0-9]{32})\b', text)
        return match.group(1) if match else None

    @staticmethod
    def extract_device(text: str) -> Optional[str]:
        """Extracts device identifier (e.g., TARA-DEVICE-002, LFAM-3D-PRINTER)."""
        match = re.search(r'\b(TARA-DEVICE-[a-zA-Z0-9_\-]+|LFAM-[a-zA-Z0-9_\-]+|DEVICE-[a-zA-Z0-9_\-]+)\b', text, re.IGNORECASE)
        return match.group(1) if match else None

    @staticmethod
    def extract_datetime(text: str) -> Optional[str]:
        """Extracts ISO date/time or relative date markers."""
        # ISO timestamp or date
        iso_match = re.search(r'\b\d{4}-\d{2}-\d{2}(?:[T\s]\d{2}:\d{2}(?::\d{2})?)?\b', text)
        if iso_match:
            return iso_match.group(0)

        # Relative date keywords
        rel_match = re.search(r'\b(today|yesterday|tomorrow|now)\b', text, re.IGNORECASE)
        if rel_match:
            return rel_match.group(1).lower()

        return None

    @staticmethod
    def extract_amounts(text: str) -> List[Dict[str, Any]]:
        """Extracts numeric values, limits, and percentages."""
        amounts = []
        # Limits: limit 5, top 10, first 3
        limit_match = re.search(r'\b(?:limit|top|first)\s*(\d+)\b', text, re.IGNORECASE)
        if limit_match:
            amounts.append({"type": "limit", "value": int(limit_match.group(1))})

        # Numbers with units (e.g. 42.5 MB, 10 lines, 100 ms)
        unit_matches = re.finditer(r'\b(\d+(?:\.\d+)?)\s*(%|percent|mb|gb|bytes|lines|tokens|ms|s|seconds|hours?)\b', text, re.IGNORECASE)
        for m in unit_matches:
            amounts.append({"type": "metric", "value": float(m.group(1)), "unit": m.group(2).lower()})

        # Standalone numbers
        if not amounts:
            num_matches = re.findall(r'\b\d+(?:\.\d+)?\b', text)
            for n in num_matches[:2]:
                val = float(n) if "." in n else int(n)
                amounts.append({"type": "number", "value": val})

        return amounts

    @classmethod
    def extract_all_slots(cls, text: str) -> Dict[str, Any]:
        """Extracts all structured entity slots from text."""
        return {
            "file_path": cls.extract_file_path(text),
            "hash": cls.extract_hash(text),
            "device_id": cls.extract_device(text),
            "datetime": cls.extract_datetime(text),
            "amounts": cls.extract_amounts(text)
        }


class SemanticIntentParser:
    """
    Parses natural language user inputs into structured intents and parameters.
    Upgrades lexical matching with paraphrase mapping while preserving safety invariants.
    """

    def __init__(self, available_skills: Optional[List[str]] = None):
        self.available_skills = available_skills or []

    def set_available_skills(self, skills: List[str]) -> None:
        self.available_skills = skills

    def parse(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        text_clean = text.strip()
        text_lower = text_clean.lower()
        ctx = dict(context or {})
        slots = SlotExtractor.extract_all_slots(text_clean)

        if "intent_override" in ctx and isinstance(ctx["intent_override"], dict):
            return dict(ctx["intent_override"])

        # Dynamic Engine Dispatch
        if text_lower.startswith("execute engine") or text_lower.startswith("run engine"):
            parts = text_clean.split(maxsplit=2)
            task_type = parts[2].strip() if len(parts) > 2 else ""
            return {
                "intent": "EXECUTE_ENGINE",
                "task_type": task_type,
                "payload": ctx.get("payload", ctx.get("params", {}))
            }

        # --------------------------------------------------------------------
        # 1. COMPOSITE MULTI-STEP INTENT CHECK
        # --------------------------------------------------------------------
        # Detect requests requiring multi-step planning (e.g. "Find X, calculate its SHA-256, and compare...")
        composite_markers = [
            r"\b(find|inspect|get|read)\b.+(?:,\s*|\s+and\s+|\s+then\s+)(?:calculate|compute|verify)\b.+(?:,\s*|\s+and\s+|\s+then\s+)(?:check|compare|diagnostics|telemetry)\b",
            r"\b(find|inspect|get|read)\b.+\b(calculate|compute|verify)\b.+\b(diagnostics|telemetry|compare|check)\b",
            r"\bfirst\b.+\bthen\b.+\bfinally\b",
            r"\bstep 1\b.+\bstep 2\b",
            r"\b(inspect|read)\b.+\band\b.+\b(calculate|compute)\b.+\bhash\b",
            r"\b(calculate|compute)\b.+\bhash\b.+\band\b.+\bcompare\b"
        ]
        for pattern in composite_markers:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return {
                    "intent": "COMPOSITE_PLAN",
                    "raw_text": text_clean,
                    "slots": slots,
                    "params": ctx
                }

        # --------------------------------------------------------------------
        # 2. DETERMINISTIC SAFETY INTERCEPTORS (Fail Closed / Invariants)
        # --------------------------------------------------------------------
        # Mandatory CSAM check
        if any(bad in text_lower for bad in ["child abuse", "csam", "child sexual"]):
            return {
                "intent": "GUARDED_ACTION",
                "action_type": "display_media",
                "params": {"topic": text_clean, "media_type": "csam_content"}
            }

        # Content Safety (Nudity / Explicit)
        if any(bad in text_lower for bad in ["nude photo", "nude video", "porn", "naked photo", "naked video"]):
            return {
                "intent": "GUARDED_ACTION",
                "action_type": "display_media",
                "params": {"topic": text_clean, "media_type": "nude_photo"}
            }

        # Guarded Deletion
        if any(w in text_lower for w in ["delete", "remove", "erase", "purge"]) and any(w in text_lower for w in ["file", "folder", "directory", "database", "db"]):
            file_target = slots.get("file_path") or ctx.get("file_path", text_clean)
            return {
                "intent": "GUARDED_ACTION",
                "action_type": "delete_file",
                "params": {
                    "file_path": file_target,
                    "is_important": ctx.get("is_important", True)
                }
            }

        # Guarded Machine / Configuration Mutation
        if any(w in text_lower for w in ["modify", "update", "change", "alter"]) and any(w in text_lower for w in ["machine", "config", "setting", "hardware"]):
            return {
                "intent": "GUARDED_ACTION",
                "action_type": "modify_machine_config",
                "params": ctx.get("params", {})
            }

        # Guarded Credential / Key Export
        if any(w in text_lower for w in ["export", "dump", "backup", "reveal", "read"]) and any(w in text_lower for w in ["private key", "private_key", "keystore", "secret", "seed"]):
            return {
                "intent": "GUARDED_ACTION",
                "action_type": "export_key",
                "params": ctx.get("params", {})
            }

        # --------------------------------------------------------------------
        # 3. CONTINUOUS LEARNING TRIGGERS
        # --------------------------------------------------------------------
        if any(p in text_lower for p in ["autonomous learning", "learn in background", "background learning"]):
            return {
                "intent": "AUTONOMOUS_LEARNING",
                "mode": ctx.get("mode", "DURATION_1_HOUR")
            }

        if any(p in text_lower for p in ["learn online", "search web", "research topic", "web search"]):
            query = re.sub(r"(?i)(learn online|search web|research topic|web search)\s*:?", "", text_clean).strip() or text_clean
            return {
                "intent": "ONLINE_LEARNING",
                "query": query
            }

        # --------------------------------------------------------------------
        # 4. PARAPHRASED TOOL INTENTS
        # --------------------------------------------------------------------
        # File Inspector paraphrases:
        file_inspector_patterns = [
            r"\b(inspect_file|file_inspector)\b",
            r"\b(check|see|look|examine)\s+(what is|what's)\s+inside\b",
            r"\b(inspect|examine|inspect contents of|view lines in|read)\s+(?:the\s+)?file\b",
            r"\bcheck\s+if\s+(?:the\s+)?file\s+exists\b",
            r"\bverify\s+(?:the\s+)?(?:contents|existence|size)\s+of\b",
            r"\b(check|see|count)\s+(?:how many\s+)?lines\b"
        ]
        if any(re.search(p, text_lower) for p in file_inspector_patterns):
            target_path = slots.get("file_path") or ctx.get("file_path", "")
            return {
                "intent": "EXECUTE_TOOL",
                "tool": "file_inspector",
                "params": {"file_path": target_path}
            }

        # Hash Verifier paraphrases:
        hash_patterns = [
            r"\b(hash_verifier|verify_hash|compute_hash)\b",
            r"\b(calculate|compute|get|generate)\b.*?\b(?:sha256|hash|checksum|digest)\b",
            r"\b(verify|check)\b.*?\b(?:integrity|hash|checksum|digest)\b",
            r"\bchecksum\s+of\b"
        ]
        if any(re.search(p, text_lower) for p in hash_patterns):
            target_path = slots.get("file_path") or ctx.get("file_path")
            target_hash = slots.get("hash") or ctx.get("expected_hash")
            params = {}
            if target_path:
                params["file_path"] = target_path
            if target_hash:
                params["expected_hash"] = target_hash
            if not target_path and not target_hash:
                params["data"] = text_clean
            return {
                "intent": "EXECUTE_TOOL",
                "tool": "hash_verifier",
                "params": params
            }

        # Knowledge Retriever paraphrases:
        knowledge_patterns = [
            r"\b(knowledge_retriever|search_knowledge)\b",
            r"\b(search|query)\s+(?:the\s+)?knowledge\s*(?:base)?\b",
            r"\b(look up|find)\s+(?:verified\s+)?(?:facts?|information)\s+about\b"
        ]
        if any(re.search(p, text_lower) for p in knowledge_patterns):
            q_text = re.sub(r"(?i)(knowledge_retriever|search knowledge base|search knowledge|find verified facts about)\s*:?", "", text_clean).strip() or text_clean
            return {
                "intent": "EXECUTE_TOOL",
                "tool": "knowledge_retriever",
                "params": {"query": q_text, "topic": ctx.get("topic")}
            }

        # Provenance Tracker paraphrases:
        if any(w in text_lower for w in ["provenance_tracker", "record provenance", "audit trail", "track provenance"]):
            return {
                "intent": "EXECUTE_TOOL",
                "tool": "provenance_tracker",
                "params": ctx.get("tool_params", {"action": "custom_audit", "target": text_clean})
            }

        # --------------------------------------------------------------------
        # 5. SEMANTIC SKILL TRIGGERS
        # --------------------------------------------------------------------
        # Diagnostics
        if any(p in text_lower for p in ["diagnostics", "system telemetry", "telemetry", "system health", "cpu load", "ram usage", "check cpu"]):
            return {
                "intent": "EXECUTE_SKILL",
                "skill": "diagnostics",
                "params": ctx.get("params", {})
            }

        # Developer
        if any(p in text_lower for p in ["check syntax", "lint code", "validate python", "syntax check"]):
            code_str = ctx.get("code", text_clean)
            return {
                "intent": "EXECUTE_SKILL",
                "skill": "developer",
                "params": {"code": code_str}
            }

        # Device
        if any(p in text_lower for p in ["gcode", "3d printer", "lfam", "printer kinematics"]):
            return {
                "intent": "EXECUTE_SKILL",
                "skill": "device",
                "params": {"gcode": ctx.get("gcode", text_clean)}
            }

        # Translation
        if any(p in text_lower for p in ["translate", "translation", "in kannada", "glossary"]):
            word_target = re.sub(r"(?i)(translate|in kannada|translation)\s*:?", "", text_clean).strip() or text_clean
            return {
                "intent": "EXECUTE_SKILL",
                "skill": "translation",
                "params": {"text": word_target}
            }

        # Dynamic skill match against registered skills
        for sk in self.available_skills:
            pattern = r"\b" + re.escape(sk.lower().replace("_", " ")) + r"\b"
            if re.search(pattern, text_lower) or sk.lower() in text_lower:
                return {
                    "intent": "EXECUTE_SKILL",
                    "skill": sk,
                    "params": ctx.get("params", {"input": text_clean})
                }

        # Audio / Video / PDF fallback keywords
        if any(w in text_lower for w in ["wav", "sound", "audio"]):
            return {"intent": "EXECUTE_SKILL", "skill": "audio", "params": {"file_path": slots.get("file_path") or ctx.get("file_path", text_clean)}}
        if any(w in text_lower for w in ["mp4", "video"]):
            return {"intent": "EXECUTE_SKILL", "skill": "video", "params": {"file_path": slots.get("file_path") or ctx.get("file_path", text_clean)}}
        if "pdf" in text_lower:
            return {"intent": "EXECUTE_SKILL", "skill": "pdf", "params": {"file_path": slots.get("file_path") or ctx.get("file_path", text_clean)}}

        # --------------------------------------------------------------------
        # 6. DYNAMIC CAPABILITY REGISTRY DISCOVERY
        # --------------------------------------------------------------------
        try:
            from tara_core.registry import CapabilityRegistry, CapabilityCategory
            registry = CapabilityRegistry.get_default()
            matched_caps = registry.match_intent(text_clean, ctx)
            if matched_caps:
                best_cap = matched_caps[0]
                if best_cap.category == CapabilityCategory.TOOL:
                    tool_params = dict(slots)
                    if "params" in ctx:
                        tool_params.update(ctx["params"])
                    return {
                        "intent": "EXECUTE_TOOL",
                        "tool": best_cap.name,
                        "params": tool_params
                    }
                elif best_cap.category == CapabilityCategory.SKILL:
                    return {
                        "intent": "EXECUTE_SKILL",
                        "skill": best_cap.name,
                        "params": ctx.get("params", {"input": text_clean})
                    }
                elif best_cap.category == CapabilityCategory.AGENT:
                    return {
                        "intent": "EXECUTE_AGENT",
                        "role": best_cap.name,
                        "params": ctx.get("params", {"input": text_clean})
                    }
                elif best_cap.category == CapabilityCategory.EXTENSION:
                    return {
                        "intent": "EXECUTE_EXTENSION",
                        "capability": best_cap.capability_id,
                        "params": ctx.get("params", {"input": text_clean})
                    }
        except Exception:
            pass

        # --------------------------------------------------------------------
        # 7. DEFAULT CONVERSATIONAL / FACTUAL QUERY
        # --------------------------------------------------------------------
        return {
            "intent": "CONVERSATIONAL",
            "query": text_clean,
            "slots": slots
        }
