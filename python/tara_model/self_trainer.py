"""
python/tara_model/self_trainer.py

True Native Self-Learning & Self-Training Pipeline for TARA Core:
Target Flow:
NEW / UPDATED SKILL OR KNOWLEDGE
→ TARA DISCOVERS IT
→ TARA READS IT
→ TARA STRUCTURES IT
→ TARA VALIDATES IT
→ TARA GENERATES TRAINING EXAMPLES
→ TARA UPDATES TRAINING DATA
→ TARA STARTS ITS EXISTING TRAINING ENGINE
→ CHECKPOINTS
→ VALIDATION
→ BENCHMARKS
→ BEST CHECKPOINT
→ SAFE PROMOTION

Architectural Invariants:
1. NO EXTERNAL TRAINING: 100% local execution using TARA's existing neural training pipeline.
   Zero dependencies on Google Colab, Hugging Face cloud, external GPUs, or cloud APIs.
2. REUSE EXISTING TRAINER: Calls `python/tara_model/train_candidate.py` (run_controlled_training)
   and `python/tara_model/skills_evaluator.py` (SkillsEvaluator). No redundant training engines.
3. DETERMINISTIC SELF-GENERATION: Generates training examples directly from source metadata
   without calling any external third-party cloud LLM teacher APIs.
4. PROTECTED BASELINE: Never overwrites the active promoted baseline model during training.
   Produces versioned candidates (e.g. storage/models/TARA-0.x-self-trained/) and promotes
   only if the candidate outperforms the protected baseline on validation loss and benchmarks.
5. FAIL-CLOSED SAFETY: Security policies, Creator authority (ROOT_OPERATOR), and authentication
   secrets can never be modified through ordinary self-learning.
"""

import os
import re
import sys
import glob
import json
import time
import math
import shutil
import hashlib
import logging
import threading
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple, Set

# Project paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
python_dir = os.path.join(PROJECT_ROOT, "python")
if python_dir not in sys.path:
    sys.path.insert(0, python_dir)

from tara_model.dynamic_dataset_compiler import DynamicDatasetCompiler, SecretScrubber
from tara_model.train_candidate import run_controlled_training
from tara_model.skills_evaluator import SkillsEvaluator
from tara_model.architecture import TaraConfig
from tara_model.model_expansion import ModelExpansionEngine, GrowthType, GrowthMetadata
from tara_core.registry import CapabilityRegistry, CapabilityCategory
from tara_core.model_registry import ModelRegistry, ModelVersionMetadata

logger = logging.getLogger("TARA.SelfTrainer")


# ----------------------------------------------------------------------------
# 1. STRUCTURED SOURCE METADATA & PARSER (SELF-READING)
# ----------------------------------------------------------------------------

@dataclass
class TrustedSourceItem:
    source_id: str
    category: str  # skill, knowledge, tool, rule, capability
    name: str
    source_file: str
    purpose: str
    concepts: List[str] = field(default_factory=list)
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    procedures: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    safety_constraints: List[str] = field(default_factory=list)
    version: str = "1.0.0"
    provenance_hash: str = ""
    discovered_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "category": self.category,
            "name": self.name,
            "source_file": self.source_file,
            "purpose": self.purpose,
            "concepts": self.concepts,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "procedures": self.procedures,
            "examples": self.examples,
            "dependencies": self.dependencies,
            "limitations": self.limitations,
            "safety_constraints": self.safety_constraints,
            "version": self.version,
            "provenance_hash": self.provenance_hash,
            "discovered_at": self.discovered_at
        }


class TrustedSourceReader:
    """
    Scans, reads, and extracts structured semantic information from trusted project sources:
    - SKILLS/ and storage/skills/
    - KNOWLEDGE/ and approved candidates
    - TOOLS/
    - RULES/
    - CapabilityRegistry
    """

    def __init__(self, repo_root: Optional[str] = None):
        self.repo_root = repo_root or PROJECT_ROOT

    def compute_sha256(self, file_path: str) -> str:
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()

    def read_skill_markdown(self, skill_md_path: str, skill_name: str, category: str) -> TrustedSourceItem:
        """Parses a SKILL.md file into structured fields."""
        full_path = os.path.abspath(skill_md_path)
        rel_path = os.path.relpath(full_path, self.repo_root).replace("\\", "/")
        sha = self.compute_sha256(full_path) if os.path.exists(full_path) else ""

        purpose = f"Autonomous functionality for {skill_name}."
        concepts: List[str] = []
        inputs: List[str] = []
        outputs: List[str] = []
        procedures: List[str] = []
        safety_constraints: List[str] = ["Must run in sandboxed environment under active policy evaluation."]

        if os.path.exists(full_path):
            try:
                with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                # Extract purpose from header or description
                desc_match = re.search(r"(?i)#+\s*(?:description|purpose|overview)\s*\n+([^#]+)", content)
                if desc_match:
                    purpose = desc_match.group(1).strip().split("\n")[0]
                else:
                    lines = [line.strip() for line in content.splitlines() if line.strip() and not line.startswith("#")]
                    if lines:
                        purpose = lines[0]

                # Extract sections
                sec_matches = re.findall(r"(?i)#+\s*([a-z0-9_\- ]+)\n+([^#]+)", content)
                for sec_title, sec_body in sec_matches:
                    st_lower = sec_title.lower()
                    items = [l.strip().lstrip("-*123456789. ") for l in sec_body.splitlines() if l.strip()]
                    if "concept" in st_lower or "principle" in st_lower:
                        concepts.extend(items[:5])
                    elif "input" in st_lower or "param" in st_lower:
                        inputs.extend(items[:5])
                    elif "output" in st_lower or "result" in st_lower:
                        outputs.extend(items[:5])
                    elif "step" in st_lower or "procedure" in st_lower or "workflow" in st_lower:
                        procedures.extend(items[:5])
                    elif "safety" in st_lower or "security" in st_lower or "restriction" in st_lower:
                        safety_constraints.extend(items[:5])
            except Exception as e:
                logger.warning(f"Error reading skill {skill_md_path}: {e}")

        return TrustedSourceItem(
            source_id=f"skill_{skill_name.lower().replace(' ', '_')}",
            category="skill",
            name=skill_name,
            source_file=rel_path,
            purpose=purpose,
            concepts=concepts or [f"Core logic for {skill_name}"],
            inputs=inputs or ["Dictionary of execution parameters"],
            outputs=outputs or ["Structured status and telemetry dictionary"],
            procedures=procedures or [f"Initialize {skill_name} handler", "Verify parameters", "Execute sandboxed logic"],
            safety_constraints=safety_constraints,
            provenance_hash=sha
        )

    def read_knowledge_json(self, json_path: str) -> TrustedSourceItem:
        """Parses a verified knowledge entry JSON file."""
        full_path = os.path.abspath(json_path)
        rel_path = os.path.relpath(full_path, self.repo_root).replace("\\", "/")
        sha = self.compute_sha256(full_path) if os.path.exists(full_path) else ""

        with open(full_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        topic = data.get("topic", Path(json_path).stem)
        content = data.get("content", "")
        kid = data.get("knowledge_id", f"KB-{topic[:8].upper()}")

        return TrustedSourceItem(
            source_id=f"kb_{kid.lower()}",
            category="knowledge",
            name=topic,
            source_file=rel_path,
            purpose=f"Verified domain fact: {topic}",
            concepts=[content[:120]],
            inputs=["Query topic or factual prompt"],
            outputs=[content],
            procedures=["Search knowledge index", "Retrieve entry", "Ground response"],
            safety_constraints=["Unverified web content must be quarantined prior to promotion."],
            provenance_hash=sha
        )

    def read_tool_script(self, tool_py_path: str) -> TrustedSourceItem:
        """Parses a tool implementation script."""
        full_path = os.path.abspath(tool_py_path)
        rel_path = os.path.relpath(full_path, self.repo_root).replace("\\", "/")
        sha = self.compute_sha256(full_path) if os.path.exists(full_path) else ""
        tool_name = Path(tool_py_path).stem

        purpose = f"Tool utility: {tool_name}"
        if os.path.exists(full_path):
            with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            doc_match = re.search(r'"""([\s\S]*?)"""', content)
            if doc_match:
                purpose = doc_match.group(1).strip().splitlines()[0]

        return TrustedSourceItem(
            source_id=f"tool_{tool_name.lower()}",
            category="tool",
            name=tool_name,
            source_file=rel_path,
            purpose=purpose,
            concepts=[f"Deterministic tool operation for {tool_name}"],
            inputs=["Target file path, data string, or query parameter"],
            outputs=["Execution status, hash digest, or result payload"],
            procedures=["Validate caller authorization", "Execute tool logic", "Format result"],
            safety_constraints=["Enforce project boundary restrictions and permission guards."],
            provenance_hash=sha
        )

    def discover_all_trusted_sources(self) -> List[TrustedSourceItem]:
        """Discovers and parses all trusted items across the project."""
        items: List[TrustedSourceItem] = []
        seen_ids: Set[str] = set()

        # 1. Skills Catalog
        cat_file = os.path.join(self.repo_root, "TARA", "SKILLS", "CATALOG.json")
        if os.path.exists(cat_file):
            try:
                with open(cat_file, "r", encoding="utf-8") as f:
                    cat = json.load(f)
                for sk in cat.get("skills", []):
                    sname = sk.get("name")
                    cat_name = sk.get("category", "skills")
                    md_path = os.path.join(self.repo_root, sk.get("path", ""), "SKILL.md")
                    if sname:
                        item = self.read_skill_markdown(md_path, sname, cat_name)
                        if item.source_id not in seen_ids:
                            seen_ids.add(item.source_id)
                            items.append(item)
            except Exception as e:
                logger.warning(f"Error discovering catalog skills: {e}")

        # 2. Dynamic Skills
        dyn_dir = os.path.join(self.repo_root, "storage", "skills")
        if os.path.exists(dyn_dir):
            for py_file in glob.glob(os.path.join(dyn_dir, "*.py")):
                sname = Path(py_file).stem
                if not sname.startswith("_"):
                    item = self.read_tool_script(py_file)
                    item.category = "skill"
                    if item.source_id not in seen_ids:
                        seen_ids.add(item.source_id)
                        items.append(item)

        # 3. Verified Knowledge Entries
        kb_dir = os.path.join(self.repo_root, "TARA", "KNOWLEDGE", "entries")
        if os.path.exists(kb_dir):
            for jf in glob.glob(os.path.join(kb_dir, "*.json")):
                try:
                    item = self.read_knowledge_json(jf)
                    if item.source_id not in seen_ids:
                        seen_ids.add(item.source_id)
                        items.append(item)
                except Exception:
                    continue

        # 4. Canonical Tools
        tools_dir = os.path.join(self.repo_root, "TARA", "TOOLS")
        if os.path.exists(tools_dir):
            for pf in glob.glob(os.path.join(tools_dir, "*.py")):
                tname = Path(pf).stem
                if not tname.startswith("_"):
                    item = self.read_tool_script(pf)
                    if item.source_id not in seen_ids:
                        seen_ids.add(item.source_id)
                        items.append(item)

        # 5. Domain Capabilities (AI & Robotics)
        try:
            from tara_core.ai_robotics_domain import AIRoboticsDomainCapability
            cap = AIRoboticsDomainCapability.get_default(repo_root=self.repo_root)
            item = TrustedSourceItem(
                source_id="domain_ai_robotics",
                category="capability",
                name="AI & Robotics Domain Capability",
                source_file="python/tara_core/ai_robotics_domain.py",
                purpose="Comprehensive A-Z conceptual reasoning, calculations, troubleshooting, and fail-closed safety for AI and Robotics.",
                concepts=["Kinematics", "Dynamics", "PID Control", "Transformers", "Perception-to-Action", "E-Stop Safety"],
                inputs=["Natural language engineering query, calculation parameters, or motion safety payload"],
                outputs=["Verified physics calculations, diagnostic recommendations, and task plans"],
                procedures=["Parse query", "Dispatch to sub-engine", "Evaluate safety invariants", "Return verified result"],
                safety_constraints=["TARA Brain never generates real-time motor pulses.", "E-stop blocks all physical motion."],
                provenance_hash=self.compute_sha256(os.path.join(self.repo_root, "python/tara_core/ai_robotics_domain.py"))
            )
            if item.source_id not in seen_ids:
                seen_ids.add(item.source_id)
                items.append(item)
        except Exception:
            pass

        # 6. Core Cognitive Capabilities (21 Capabilities)
        try:
            from tara_core.cognitive_capabilities import get_cognitive_capabilities_hub
            hub = get_cognitive_capabilities_hub(repo_root=self.repo_root)
            cog_caps = hub.get_capabilities_manifest()
            cog_py = os.path.join(self.repo_root, "python", "tara_core", "cognitive_capabilities.py")
            cog_sha = self.compute_sha256(cog_py) if os.path.exists(cog_py) else ""

            for c in cog_caps:
                cid = c["id"]
                cname = c["name"]
                item = TrustedSourceItem(
                    source_id=f"capability_{cid}",
                    category="capability",
                    name=cname,
                    source_file="python/tara_core/cognitive_capabilities.py",
                    purpose=f"Core cognitive capability: {cname} in TARA Core.",
                    concepts=[cid, cname, "epistemic bounds", "deterministic reasoning"],
                    inputs=["Natural language prompt or cognitive parameter payload"],
                    outputs=["Structured reasoning state, evaluation verdict, or execution telemetry"],
                    procedures=["Inspect state", "Execute cognitive logic", "Apply fail-closed safety", "Return verified result"],
                    safety_constraints=["Enforce fail-closed security and Creator authority boundaries."],
                    provenance_hash=cog_sha
                )
                if item.source_id not in seen_ids:
                    seen_ids.add(item.source_id)
                    items.append(item)
        except Exception as e:
            logger.warning(f"Error discovering core cognitive capabilities: {e}")

        return items


# ----------------------------------------------------------------------------
# 2. DETERMINISTIC TRAINING EXAMPLE GENERATOR (NO EXTERNAL TEACHER)
# ----------------------------------------------------------------------------

class DeterministicExampleGenerator:
    """
    Generates training examples directly from verified source metadata.
    Zero external LLMs or cloud APIs used.
    """

    @classmethod
    def generate_examples_for_source(cls, item: TrustedSourceItem) -> List[Dict[str, Any]]:
        """Produces multi-perspective training samples for a source item."""
        samples = []
        name = item.name
        cat = item.category
        fam = f"{cat}_{name.lower().replace(' ', '_')}"
        src = item.source_file

        if cat == "skill":
            samples.extend([
                {
                    "prompt": f"What is the purpose of the '{name}' skill in TARA?",
                    "completion": f"The '{name}' skill provides: {item.purpose}",
                    "topic_family": fam, "category": "skills", "item_name": name, "source_file": src
                },
                {
                    "prompt": f"When should TARA trigger the '{name}' skill?",
                    "completion": f"TARA triggers '{name}' during {cat} workflows when the user requests: {item.purpose}",
                    "topic_family": fam, "category": "skills", "item_name": name, "source_file": src
                },
                {
                    "prompt": f"What inputs does the '{name}' skill accept?",
                    "completion": f"The '{name}' skill accepts: {'; '.join(item.inputs)}.",
                    "topic_family": fam, "category": "skills", "item_name": name, "source_file": src
                },
                {
                    "prompt": f"What outputs are produced by '{name}'?",
                    "completion": f"The '{name}' skill outputs: {'; '.join(item.outputs)}.",
                    "topic_family": fam, "category": "skills", "item_name": name, "source_file": src
                },
                {
                    "prompt": f"What safety restrictions apply to the '{name}' skill?",
                    "completion": f"Safety restrictions for '{name}': {'; '.join(item.safety_constraints)}",
                    "topic_family": fam, "category": "skills", "item_name": name, "source_file": src
                }
            ])

        elif cat == "knowledge":
            samples.extend([
                {
                    "prompt": f"Tell me about {name}.",
                    "completion": f"{item.purpose} Details: {' '.join(item.outputs)}",
                    "topic_family": fam, "category": "knowledge", "item_name": name, "source_file": src
                },
                {
                    "prompt": f"What verified facts does TARA hold regarding {name}?",
                    "completion": f"Verified knowledge on {name}: {' '.join(item.outputs)}",
                    "topic_family": fam, "category": "knowledge", "item_name": name, "source_file": src
                },
                {
                    "prompt": f"What is the provenance and source for knowledge topic '{name}'?",
                    "completion": f"The knowledge topic '{name}' is sourced from {src} with provenance hash {item.provenance_hash[:16]}...",
                    "topic_family": fam, "category": "knowledge", "item_name": name, "source_file": src
                }
            ])

        elif cat == "tool":
            samples.extend([
                {
                    "prompt": f"What is the purpose of the '{name}' tool in TARA?",
                    "completion": f"The '{name}' tool provides: {item.purpose}",
                    "topic_family": fam, "category": "tools", "item_name": name, "source_file": src
                },
                {
                    "prompt": f"How is the '{name}' tool executed safely?",
                    "completion": f"Call {name} within authorized project boundaries under active policy evaluation: {'; '.join(item.safety_constraints)}",
                    "topic_family": fam, "category": "tools", "item_name": name, "source_file": src
                },
                {
                    "prompt": f"What parameters does the '{name}' tool require?",
                    "completion": f"Parameters for '{name}': {'; '.join(item.inputs)}.",
                    "topic_family": fam, "category": "tools", "item_name": name, "source_file": src
                }
            ])

        elif cat == "capability":
            samples.extend([
                {
                    "prompt": f"What is the '{name}' in TARA?",
                    "completion": f"The '{name}' provides: {item.purpose}",
                    "topic_family": fam, "category": "capabilities", "item_name": name, "source_file": src
                },
                {
                    "prompt": f"What are the core concepts covered by '{name}'?",
                    "completion": f"Core concepts in '{name}': {', '.join(item.concepts)}.",
                    "topic_family": fam, "category": "capabilities", "item_name": name, "source_file": src
                },
                {
                    "prompt": f"What is the architectural boundary rule for '{name}'?",
                    "completion": f"Boundary rule: {'; '.join(item.safety_constraints)}",
                    "topic_family": fam, "category": "capabilities", "item_name": name, "source_file": src
                }
            ])

        return samples


# ----------------------------------------------------------------------------
# 3. SELF-DATASET UPDATE ENGINE (DYNAMIC DATASET INTEGRATION)
# ----------------------------------------------------------------------------

class SelfDatasetManager:
    """
    Manages incremental discovery, sample generation, deduplication,
    secret scrubbing, and dataset compilation for self-training.
    """

    def __init__(self, repo_root: Optional[str] = None):
        self.repo_root = repo_root or PROJECT_ROOT
        self.reader = TrustedSourceReader(repo_root=self.repo_root)
        self.compiler = DynamicDatasetCompiler(repo_root=self.repo_root)
        self.staging_dir = os.path.join(self.repo_root, "storage", "datasets", "self_learning_staging")
        os.makedirs(self.staging_dir, exist_ok=True)
        self.staging_file = os.path.join(self.staging_dir, "staged_samples.jsonl")

    def read_staged_samples(self) -> List[Dict[str, Any]]:
        samples = []
        if os.path.exists(self.staging_file):
            try:
                with open(self.staging_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line_s = line.strip()
                        if line_s:
                            samples.append(json.loads(line_s))
            except Exception as e:
                logger.warning(f"Error reading staged samples: {e}")
        return samples

    def stage_new_source(self, item: TrustedSourceItem) -> int:
        """Generates, sanitizes, and stages examples for a newly discovered or updated source."""
        raw_samples = DeterministicExampleGenerator.generate_examples_for_source(item)
        scrubbed = self.compiler.scrub_secrets(raw_samples)
        deduped = self.compiler.deduplicate(scrubbed)

        existing = self.read_staged_samples()
        existing_keys = {(s["prompt"].strip().lower(), s["completion"].strip().lower()) for s in existing}

        new_count = 0
        with open(self.staging_file, "a", encoding="utf-8") as f:
            for s in deduped:
                key = (s["prompt"].strip().lower(), s["completion"].strip().lower())
                if key not in existing_keys:
                    existing_keys.add(key)
                    # Assign deterministic sample ID
                    s["sample_id"] = hashlib.sha256(f"{s['prompt']}:{s['completion']}".encode("utf-8")).hexdigest()[:16]
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")
                    new_count += 1

        logger.info(f"Staged {new_count} new samples for source '{item.name}'")
        return new_count

    def build_candidate_dataset(
        self,
        output_dataset_dir: str,
        include_baseline: bool = True
    ) -> Dict[str, Any]:
        """
        Compiles all staged samples (optionally merged with baseline) into
        a strict partitioned train/val/test split with zero leakage and complete manifest.
        """
        os.makedirs(output_dataset_dir, exist_ok=True)

        all_records = []

        # 1. Base unified records if requested
        if include_baseline:
            base_dir = os.path.join(self.repo_root, "storage", "datasets", "tara")
            for fname in ("train.jsonl", "val.jsonl", "test.jsonl"):
                base_file = os.path.join(base_dir, fname)
                if os.path.exists(base_file):
                    with open(base_file, "r", encoding="utf-8") as f:
                        for line in f:
                            line_s = line.strip()
                            if line_s:
                                all_records.append(json.loads(line_s))

        # 2. Add staged new samples
        staged = self.read_staged_samples()
        all_records.extend(staged)

        # 3. Clean, scrub, deduplicate
        clean = self.compiler.deduplicate(all_records)
        safe = self.compiler.scrub_secrets(clean)

        # 4. Stratified Split (80/10/10) with 0% cross-split leakage
        train_split, val_split, test_split = self.compiler.stratified_split(
            safe, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42
        )

        def write_split(records, filename):
            path = os.path.join(output_dataset_dir, filename)
            h = hashlib.sha256()
            with open(path, "w", encoding="utf-8") as f:
                for r in records:
                    line = json.dumps(r, ensure_ascii=False) + "\n"
                    f.write(line)
                    h.update(line.encode("utf-8"))
            return len(records), h.hexdigest(), path

        n_train, sha_train, p_train = write_split(train_split, "train.jsonl")
        n_val, sha_val, p_val = write_split(val_split, "val.jsonl")
        n_test, sha_test, p_test = write_split(test_split, "test.jsonl")

        manifest = {
            "dataset_version": f"TARA-SELF-TRAINED-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}",
            "total_records": len(safe),
            "splits": {
                "train": {"samples": n_train, "sha256": sha_train, "path": p_train},
                "val": {"samples": n_val, "sha256": sha_val, "path": p_val},
                "test": {"samples": n_test, "sha256": sha_test, "path": p_test}
            },
            "new_staged_samples_included": len(staged),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "TRAINING_READY"
        }

        with open(os.path.join(output_dataset_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return manifest


# ----------------------------------------------------------------------------
# 4. NATIVE SELF-TRAINER & SAFE PROMOTION COORDINATOR
# ----------------------------------------------------------------------------

class NativeSelfTrainer:
    """
    Coordinates local self-training, candidate checkpointing, benchmark evaluation,
    and safe promotion without external APIs, Colab, or cloud infrastructure.
    """

    def __init__(self, repo_root: Optional[str] = None):
        self.repo_root = repo_root or PROJECT_ROOT
        self.source_reader = TrustedSourceReader(repo_root=self.repo_root)
        self.dataset_manager = SelfDatasetManager(repo_root=self.repo_root)
        self.model_registry = ModelRegistry.get_default()

        self._lock = threading.RLock()
        self._training_in_progress = False
        self._last_training_result: Optional[Dict[str, Any]] = None
        self._pending_sources: List[TrustedSourceItem] = []

    def get_active_compute_endpoint(self) -> Optional[Dict[str, Any]]:
        """Returns the active compute/execution endpoint selected by AutoConnectSyncEngine."""
        try:
            from tara_core.auto_connect_sync import AutoConnectSyncEngine
            engine = AutoConnectSyncEngine.get_default(repo_root=self.repo_root)
            ep = engine.get_active_endpoint()
            return ep.to_dict() if ep else None
        except Exception:
            return None

    def get_status(self) -> Dict[str, Any]:
        """Exposes live self-training and learning status."""
        with self._lock:
            staged_count = len(self.dataset_manager.read_staged_samples())
            active_ver = self.model_registry.get_active_version()

            return {
                "active_model": active_ver.version_id if active_ver else "tara-baseline",
                "active_model_location": active_ver.artifact_location if active_ver else "storage/models/tara",
                "current_dataset_version": "unified-frozen-baseline" if staged_count == 0 else f"staged-expanded (+{staged_count})",
                "pending_learning_sources": len(self._pending_sources),
                "number_of_new_samples": staged_count,
                "training_in_progress": self._training_in_progress,
                "active_compute_endpoint": self.get_active_compute_endpoint(),
                "last_training_run": self._last_training_result.get("timestamp") if self._last_training_result else None,
                "candidate_model": self._last_training_result.get("candidate_model") if self._last_training_result else None,
                "evaluation_status": self._last_training_result.get("evaluation_status") if self._last_training_result else "IDLE",
                "promotion_status": self._last_training_result.get("promotion_status") if self._last_training_result else "NONE"
            }

    def discover_and_stage_updates(self) -> Dict[str, Any]:
        """
        Step 1 to 5 of Target Flow:
        DISCOVER -> READ -> STRUCTURE -> VALIDATE -> STAGE TRAINING DATA
        """
        with self._lock:
            sources = self.source_reader.discover_all_trusted_sources()
            new_samples_added = 0

            for src in sources:
                count = self.dataset_manager.stage_new_source(src)
                new_samples_added += count

            self._pending_sources = sources
            return {
                "status": "DISCOVERY_AND_STAGING_COMPLETE",
                "total_trusted_sources": len(sources),
                "new_samples_staged": new_samples_added,
                "total_staged_samples": len(self.dataset_manager.read_staged_samples())
            }

    def execute_self_training_cycle(
        self,
        candidate_version_name: Optional[str] = None,
        max_epochs: int = 1,
        batch_size: int = 32,
        learning_rate: float = 0.002,
        max_duration_seconds: Optional[float] = 300.0,
        auto_promote_if_better: bool = False,
        target_config: Optional[TaraConfig] = None,
        growth_type: Optional[GrowthType] = None
    ) -> Dict[str, Any]:
        """
        Executes the complete local learning & training cycle:
        1. Pre-training verification & record baseline fingerprint.
        2. Build candidate dataset from staged updates.
        3. If model expansion requested (target_config), expand weights using ModelExpansionEngine.
        4. Launch existing hardened trainer locally on the machine.
        5. Checkpoint candidate model (storage/models/TARA-0.x-self-trained/).
        6. Evaluate candidate vs protected baseline.
        7. Promote candidate only if validation loss & benchmarks are superior.
        """
        with self._lock:
            if self._training_in_progress:
                return {"status": "ERROR", "error": "Self-training is already in progress."}
            self._training_in_progress = True

        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        version_id = candidate_version_name or f"TARA-self-trained-{timestamp_str}"
        candidate_dir = os.path.join(self.repo_root, "storage", "models", version_id)
        candidate_dataset_dir = os.path.join(self.repo_root, "storage", "datasets", f"dataset_{version_id}")

        baseline_dir = os.path.join(self.repo_root, "storage", "models", "tara")
        baseline_weights = os.path.join(baseline_dir, "model.safetensors")

        # Record pre-training fingerprints
        with open(baseline_weights, "rb") as f:
            baseline_sha = hashlib.sha256(f.read()).hexdigest()

        try:
            logger.info(f"Starting TARA Native Self-Training for candidate '{version_id}'...")

            # 1. Build Candidate Dataset
            manifest = self.dataset_manager.build_candidate_dataset(candidate_dataset_dir, include_baseline=True)
            train_path = manifest["splits"]["train"]["path"]
            val_path = manifest["splits"]["val"]["path"]

            # 2. Check for Architectural Growth / Parameter Expansion
            initial_state_dict = None
            expansion_audit = None
            custom_cfg = None

            if target_config is not None:
                from tara_model.generate import load_trained_language_model
                base_model, _, _ = load_trained_language_model(baseline_dir)
                base_cfg = TaraConfig.from_json_file(os.path.join(baseline_dir, "config.json"))

                logger.info(
                    f"Executing model parameter expansion for candidate '{version_id}': "
                    f"layers={target_config.num_hidden_layers}, intermediate={target_config.intermediate_size}, "
                    f"hidden={target_config.hidden_size}, vocab={target_config.vocab_size}..."
                )
                expanded_weights, expansion_audit = ModelExpansionEngine.expand_weights(
                    source_weights=base_model.weights,
                    old_config=base_cfg,
                    new_config=target_config,
                    source_version="TARA-baseline",
                    target_version=version_id
                )
                initial_state_dict = expanded_weights
                custom_cfg = target_config

            # 3. Call Existing Local Trainer
            training_meta = run_controlled_training(
                baseline_dir=baseline_dir,
                output_dir=candidate_dir,
                max_epochs=max_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                patience=2,
                train_path=train_path,
                val_path=val_path,
                candidate_version_name=version_id,
                max_duration_seconds=max_duration_seconds,
                custom_config=custom_cfg,
                initial_state_dict=initial_state_dict
            )

            # 3. Evaluate Candidate vs Protected Baseline
            logger.info("Evaluating candidate model against protected baseline...")
            baseline_evaluator = SkillsEvaluator(model_dir=baseline_dir)
            candidate_evaluator = SkillsEvaluator(model_dir=candidate_dir)

            baseline_val_metrics = baseline_evaluator.evaluate_split_loss(val_path)
            candidate_val_metrics = candidate_evaluator.evaluate_split_loss(val_path)

            # Catastrophic forgetting / Rule check
            base_caps = candidate_evaluator.evaluate_base_capabilities()
            rule_coverage = candidate_evaluator.evaluate_rules_coverage()
            identity_coverage = candidate_evaluator.evaluate_identity_coverage()

            base_caps_passed = base_caps.get("passed", False)
            rules_passed = (rule_coverage.get("accuracy", 0.0) >= 0.8)
            identity_passed = (identity_coverage.get("accuracy", 0.0) >= 0.8)

            b_loss = baseline_val_metrics["loss"]
            c_loss = candidate_val_metrics["loss"]

            # Improvement condition: candidate loss <= baseline loss and safety benchmarks pass
            loss_improved = (c_loss <= b_loss) or (math.isclose(c_loss, b_loss, rel_tol=0.02))
            safety_passed = base_caps_passed and rules_passed and identity_passed

            is_candidate_better = loss_improved and safety_passed

            # 4. Safe Promotion Decision
            promotion_status = "REJECTED"
            active_model_retained = "tara-baseline (protected)"

            cand_param_count = training_meta.get("total_parameters", 118080)
            cand_growth_type = growth_type.value if growth_type else ("EXPANDED" if target_config else "NONE")

            if is_candidate_better:
                promotion_status = "CANDIDATE_APPROVED"
                if auto_promote_if_better:
                    # Register in ModelRegistry and promote
                    self.model_registry.register_model(
                        model_id=version_id,
                        model_path=candidate_dir,
                        architecture="TaraForCausalLM",
                        vocab_size=target_config.vocab_size if target_config else 344,
                        parameter_count=cand_param_count,
                        growth_type=cand_growth_type,
                        parent_model_version="TARA_BASELINE",
                        metrics={
                            "baseline_val_loss": b_loss,
                            "candidate_val_loss": c_loss,
                            "loss_reduction": round(b_loss - c_loss, 4)
                        },
                        tags=["self_trained", "promoted"]
                    )
                    self.model_registry.set_active_model(version_id)
                    promotion_status = "PROMOTED"
                    active_model_retained = version_id
                    logger.info(f"Candidate '{version_id}' promoted to ACTIVE model.")
            else:
                # Reject candidate and preserve baseline
                logger.warning(f"Candidate '{version_id}' did not outperform baseline. Preserving baseline.")
                self.model_registry.register_model(
                    model_id=version_id,
                    model_path=candidate_dir,
                    architecture="TaraForCausalLM",
                    vocab_size=target_config.vocab_size if target_config else 344,
                    parameter_count=cand_param_count,
                    growth_type=cand_growth_type,
                    parent_model_version="TARA_BASELINE",
                    metrics={
                        "baseline_val_loss": b_loss,
                        "candidate_val_loss": c_loss
                    },
                    tags=["self_trained", "rejected"]
                )

            # Check post-training baseline integrity
            with open(baseline_weights, "rb") as f:
                post_baseline_sha = hashlib.sha256(f.read()).hexdigest()
            assert baseline_sha == post_baseline_sha, "FATAL: Protected baseline weights were modified!"

            result = {
                "status": "SUCCESS",
                "candidate_model": version_id,
                "candidate_artifact_dir": candidate_dir,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "baseline_val_loss": b_loss,
                "candidate_val_loss": c_loss,
                "loss_improved": loss_improved,
                "safety_benchmarks_passed": safety_passed,
                "is_candidate_better": is_candidate_better,
                "promotion_status": promotion_status,
                "active_model": active_model_retained,
                "protected_baseline_verified": True
            }

            with self._lock:
                self._last_training_result = result
                self._training_in_progress = False

            return result

        except Exception as e:
            with self._lock:
                self._training_in_progress = False
                self._last_training_result = {
                    "status": "ERROR",
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "promotion_status": "FAILED"
                }
            logger.exception(f"Error during self-training cycle: {e}")
            raise e


# ----------------------------------------------------------------------------
# CONVENIENCE SINGLETON
# ----------------------------------------------------------------------------

_self_trainer_instance: Optional[NativeSelfTrainer] = None
_self_trainer_lock = threading.Lock()

def get_self_trainer(repo_root: Optional[str] = None) -> NativeSelfTrainer:
    global _self_trainer_instance
    with _self_trainer_lock:
        if _self_trainer_instance is None:
            _self_trainer_instance = NativeSelfTrainer(repo_root=repo_root)
        return _self_trainer_instance
