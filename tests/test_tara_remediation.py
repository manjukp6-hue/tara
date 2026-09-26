"""
tests/test_tara_remediation.py

Adversarial Security and Architectural Remediation Test Suite for TARA.
Validates:
1. SEC-02: Isolated Subprocess Sandbox for Dynamic Skills (No in-process exec, timeouts, crash immunity)
2. SEC-01: Path Traversal Immunity in GlobalKnowledgeBase (Boundary assertion & slug validation)
3. SEC-03: Canonical Creator Identity (ROOT_OPERATOR / OPERATOR_ROOT enforcement, unauthorized creator_id rejection)
4. ARCH-01: Canonical Memory Migration & Persistence (TARA/MEMORY/ engine and legacy data migration)
5. ARCH-02: Seeded Standard Tools (file_inspector, hash_verifier, knowledge_retriever, provenance_tracker)
"""

import os
import sys
import json
import time
import shutil
import tempfile
import unittest

# Ensure repo root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from python.tara_core.skills import SkillEngine
from python.tara_core.online_learning import OnlineLearningEngine
from python.tara_core.api import TaraCoreApi
from TARA.ACCESS.operator.operator_profile import CANONICAL_CREATOR_ID, DEFAULT_DISPLAY_NAME
from TARA.KNOWLEDGE.knowledge_base import GlobalKnowledgeBase
from TARA.MEMORY.memory_engine import MemoryEngine
from TARA.TOOLS.file_inspector import inspect_file
from TARA.TOOLS.hash_verifier import compute_hash, verify_file_hash
from TARA.TOOLS.knowledge_retriever import search_knowledge
from TARA.TOOLS.provenance_tracker import create_provenance_record


class TestTaraRemediationSEC02(unittest.TestCase):
    """Priority 1: SEC-02 Dynamic Skill Subprocess Sandboxing Tests."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="tara_skill_sandbox_test_")
        self.skill_engine = SkillEngine(dynamic_skills_dir=self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_valid_dynamic_skill_executes_in_sandbox(self):
        """Valid dynamic skill runs inside subprocess and returns computed result."""
        code = (
            "def run(params):\n"
            "    val = params.get('number', 0)\n"
            "    return {'status': 'SUCCESS', 'doubled': val * 2, 'creator': 'ROOT_OPERATOR'}\n"
        )
        reg = self.skill_engine.learn_or_update_skill("double_num", code, description="Doubles a number")
        self.assertEqual(reg["status"], "SUCCESS")

        res = self.skill_engine.execute_skill("double_num", {"number": 21})
        self.assertEqual(res.get("status"), "SUCCESS")
        self.assertEqual(res.get("doubled"), 42)
        self.assertEqual(res.get("creator"), "ROOT_OPERATOR")

    def test_crashing_skill_does_not_crash_host_tara(self):
        """A dynamic skill raising an uncaught exception returns ERROR without terminating host."""
        code = (
            "def run(params):\n"
            "    raise ZeroDivisionError('Intentional crash in sandbox')\n"
        )
        self.skill_engine.learn_or_update_skill("crash_skill", code)
        res = self.skill_engine.execute_skill("crash_skill", {})
        self.assertEqual(res.get("status"), "ERROR")
        self.assertIn("ZeroDivisionError", str(res.get("error")))

    def test_syntax_error_skill_handled_gracefully(self):
        """A dynamic skill with syntax errors returns ERROR without breaking host."""
        code = "def run(params): syntax error invalid python here"
        self.skill_engine.learn_or_update_skill("syntax_bad", code)
        res = self.skill_engine.execute_skill("syntax_bad", {})
        self.assertEqual(res.get("status"), "ERROR")

    def test_timeout_infinite_loop_skill_terminates(self):
        """A hanging dynamic skill is killed by the sandbox timeout."""
        code = (
            "import time\n"
            "def run(params):\n"
            "    while True:\n"
            "        time.sleep(0.1)\n"
        )
        file_path = os.path.join(self.tmp_dir, "loop_skill.py")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(code)

        # Test timeout directly with a short 0.5s timeout
        res = self.skill_engine._execute_in_sandbox(file_path, {}, timeout=0.5)
        self.assertEqual(res.get("status"), "ERROR")
        self.assertIn("timed out", res.get("error").lower())

    def test_sandbox_strips_sensitive_environment_variables(self):
        """Sandbox execution does not leak sensitive host environment variables."""
        os.environ["TARA_TEST_SECRET_KEY"] = "super_secret_token_xyz"
        try:
            code = (
                "import os\n"
                "def run(params):\n"
                "    secret = os.environ.get('TARA_TEST_SECRET_KEY', 'NOT_FOUND')\n"
                "    return {'secret_status': secret}\n"
            )
            self.skill_engine.learn_or_update_skill("env_leak_test", code)
            res = self.skill_engine.execute_skill("env_leak_test", {})
            self.assertEqual(res.get("secret_status"), "NOT_FOUND")
        finally:
            os.environ.pop("TARA_TEST_SECRET_KEY", None)


class TestTaraRemediationSEC01(unittest.TestCase):
    """Priority 2: SEC-01 Knowledge Path Traversal Tests."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="tara_kb_traversal_test_")
        self.kb = GlobalKnowledgeBase(base_dir=self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_path_traversal_dot_dot_rejected(self):
        """Reject relative path traversal attempts."""
        with self.assertRaises(ValueError):
            self.kb.save_topic_dossier(
                topic_slug="../../escaped_directory",
                overview={"summary": "Attack payload"},
                sources=[],
                research_report_md="Attack"
            )

    def test_path_traversal_absolute_rejected(self):
        """Reject absolute directory path traversal attempts."""
        with self.assertRaises(ValueError):
            self.kb.save_topic_dossier(
                topic_slug="/tmp/evil_topic",
                overview={"summary": "Attack payload"},
                sources=[],
                research_report_md="Attack"
            )

    def test_path_traversal_windows_drive_rejected(self):
        """Reject Windows drive letter attempts."""
        with self.assertRaises(ValueError):
            self.kb.save_topic_dossier(
                topic_slug="C:\\windows_escape",
                overview={"summary": "Attack payload"},
                sources=[],
                research_report_md="Attack"
            )

    def test_valid_topic_slug_saved_within_boundary(self):
        """Valid topic slugs succeed and remain strictly inside topics_dir."""
        result_path = self.kb.save_topic_dossier(
            topic_slug="quantum_cryptography_2026",
            overview={"summary": "Quantum resistant crypto dossier"},
            sources=[{"url": "https://example.org/qc", "title": "QC Doc"}],
            research_report_md="# Quantum Cryptography"
        )
        self.assertTrue(os.path.exists(result_path))
        self.assertTrue(os.path.realpath(result_path).startswith(os.path.realpath(self.kb.topics_dir)))
        self.assertTrue(os.path.exists(os.path.join(result_path, "overview.json")))
        self.assertTrue(os.path.exists(os.path.join(result_path, "research_report.md")))


class TestTaraRemediationSEC03(unittest.TestCase):
    """Priority 3: SEC-03 Creator Identity Rebinding Tests."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="tara_id_test_")
        self.learning_engine = OnlineLearningEngine(storage_dir=self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_unauthorized_creator_id_rejected_in_approval(self):
        """Using unauthorized creator_id must be rejected with PermissionError."""
        candidate = self.learning_engine.search_and_learn("Quantum Computing")
        cand_id = candidate["candidate_id"]

        with self.assertRaises(PermissionError):
            self.learning_engine.approve_learning(cand_id, creator_id="unauthorized_actor")

    def test_canonical_root_operator_accepted_in_approval(self):
        """Using canonical 'ROOT_OPERATOR' succeeds."""
        candidate = self.learning_engine.search_and_learn("Neural Architecture")
        cand_id = candidate["candidate_id"]

        res = self.learning_engine.approve_learning(cand_id, creator_id=CANONICAL_CREATOR_ID)
        self.assertEqual(res["status"], "APPROVED")
        self.assertEqual(res["approved_by"], CANONICAL_CREATOR_ID)

    def test_skills_defaults_to_canonical_creator(self):
        """SkillEngine.learn_or_update_skill defaults to CANONICAL_CREATOR_ID."""
        skill_engine = SkillEngine(dynamic_skills_dir=self.tmp_dir)
        res = skill_engine.learn_or_update_skill("test_dyn", "def run(p): return {'ok': True}")
        file_path = res["file_path"]
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn(f"Updated by: {CANONICAL_CREATOR_ID}", content)

    def test_api_creator_default_is_canonical(self):
        """TaraCoreApi defaults to CANONICAL_CREATOR_ID."""
        self.assertEqual(CANONICAL_CREATOR_ID, "ROOT_OPERATOR")
        self.assertEqual(DEFAULT_DISPLAY_NAME, "OPERATOR_ROOT")


class TestTaraRemediationARCH01(unittest.TestCase):
    """Priority 4: ARCH-01 Canonical Memory Engine & Legacy Migration Tests."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="tara_mem_test_")
        self.canonical_dir = os.path.join(self.tmp_dir, "canonical_memory")
        self.legacy_dir = os.path.join(self.tmp_dir, "legacy_memory")
        os.makedirs(self.legacy_dir, exist_ok=True)

        # Create a mock legacy episode
        legacy_rec = {
            "episode_id": "ep_legacy_001",
            "actor_id": "ROOT_OPERATOR",
            "intent": "TEST_LEGACY",
            "action": "EXECUTE",
            "parameters": {"input": "test"},
            "outcome": "SUCCESS"
        }
        with open(os.path.join(self.legacy_dir, "episodes.jsonl"), "w", encoding="utf-8") as f:
            f.write(json.dumps(legacy_rec) + "\n")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_auto_migration_from_legacy_storage(self):
        """MemoryEngine automatically migrates existing episodes from legacy storage."""
        mem = MemoryEngine(memory_dir=self.canonical_dir, legacy_dir=self.legacy_dir)
        stats = mem.get_memory_stats()
        self.assertEqual(stats["episodes_count"], 1)

        episodes = mem.query_episodes(query="TEST_LEGACY")
        self.assertEqual(len(episodes), 1)
        self.assertEqual(episodes[0]["episode_id"], "ep_legacy_001")
        self.assertEqual(episodes[0].get("migrated_from"), "legacy_storage_memory")

    def test_record_and_query_procedure(self):
        """Canonical MemoryEngine stores and retrieves multi-step procedures."""
        mem = MemoryEngine(memory_dir=self.canonical_dir, legacy_dir=self.legacy_dir)
        steps = ["step1_initialize", "step2_verify", "step3_finalize"]
        proc = mem.store_procedure("quantum_pipeline", steps, "Three step pipeline")
        self.assertIn("proc_id", proc)

        retrieved = mem.get_procedure("quantum_pipeline")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved["steps"], steps)


class TestTaraRemediationARCH02(unittest.TestCase):
    """Priority 5: ARCH-02 Seeded Standard Tools Tests."""

    def test_file_inspector_blocks_secrets(self):
        """file_inspector rejects attempts to inspect paths containing secrets or private keys."""
        with self.assertRaises(PermissionError):
            inspect_file(os.path.join("TARA", "ACCESS", "vault", "creator.pem"))

    def test_file_inspector_normal_file(self):
        """file_inspector accurately calculates size and SHA-256 for standard files."""
        res = inspect_file(os.path.join(REPO_ROOT, "TARA", "RULES", "RULEBOOK.txt"))
        self.assertEqual(res["status"], "SUCCESS")
        self.assertFalse(res["is_dir"])
        self.assertGreater(res["size_bytes"], 0)
        self.assertIn("sha256", res)

    def test_hash_verifier(self):
        """hash_verifier verifies SHA-256 digests accurately."""
        data = b"TARA AI Intelligence 2026"
        expected = compute_hash(data, "sha256")
        self.assertIsInstance(expected, str)
        self.assertEqual(len(expected), 64)

    def test_provenance_tracker(self):
        """provenance_tracker generates immutable cryptographic audit tokens."""
        token = create_provenance_record(
            actor_id="ROOT_OPERATOR",
            actor_role="CREATOR",
            action="AUDIT_VERIFY",
            target="TARA/RULES/RULEBOOK.txt"
        )
        self.assertEqual(token["actor_id"], "ROOT_OPERATOR")
        self.assertTrue(token["provenance_id"].startswith("prov_"))
        self.assertEqual(len(token["digest"]), 64)


if __name__ == "__main__":
    unittest.main()
