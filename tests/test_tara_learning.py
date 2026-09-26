"""
tests/test_tara_learning.py

Comprehensive test suite verifying the TARA Online Learning and Research System.

Canonical Creator Identity:
- CREATOR_ID = "ROOT_OPERATOR"
- CREATOR_DISPLAY_NAME = "OPERATOR_ROOT"

Verifies:
1. Single common Global Knowledge Base (TARA/KNOWLEDGE/) - no separate user/creator stores
2. Normal user web search and factual knowledge extraction
3. Creator-only hidden Learn Online controls (hidden from user tool discovery)
4. All duration options (1h, 2h, 6h, 12h, 24h) and LOOP mode
5. Automatic stop after duration expiry
6. LOOP mode continuous scheduling with duplicate suppression
7. Creator-directed deep topic research & subtopic decomposition
8. Topic directory generation (TARA/KNOWLEDGE/topics/<topic>/)
9. Source provenance, versioning, and citation tracking
10. Duplicate detection & non-duplication
11. Knowledge update/merge with update history preservation
12. Conflicting sources handling & contradictory status tagging
13. Capability synthesis (generating and testing a new tool)
14. Creator cryptographic authorization & signature enforcement
15. Unauthorized user/device rejection & security audit logging
16. Rulebook immutability & protection from automated learning
17. Persistence across application restart
18. Offline / network failure resilience & graceful recovery
"""

import os
import sys
import json
import time
import shutil
import tempfile
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from TARA.ACCESS.crypto.ed25519 import Ed25519
from TARA.ACCESS.operator.operator_profile import CANONICAL_CREATOR_ID, DEFAULT_DISPLAY_NAME
from TARA.KNOWLEDGE.knowledge_base import GlobalKnowledgeBase
from TARA.LEARNING.security_guard import LearningSecurityGuard
from TARA.LEARNING.user_search_learner import UserSearchLearner
from TARA.LEARNING.autonomous_learner import AutonomousOnlineLearner, LearningMode, DURATION_SECONDS_MAP
from TARA.LEARNING.topic_researcher import TopicResearcher
from TARA.LEARNING.capability_synthesizer import CapabilitySynthesizer
from TARA.RULES.engine.execution_guard import ExecutionGuard

class TestTaraOnlineLearningSystem(unittest.TestCase):
    """Exhaustive test suite for TARA Online Learning and Research System."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="tara_learn_test_")
        self.knowledge_dir = os.path.join(self.temp_dir, "KNOWLEDGE")
        self.learning_dir = os.path.join(self.temp_dir, "LEARNING")
        self.tools_dir = os.path.join(self.temp_dir, "TOOLS")

        os.makedirs(self.knowledge_dir, exist_ok=True)
        os.makedirs(self.learning_dir, exist_ok=True)
        os.makedirs(self.tools_dir, exist_ok=True)

        self.creator_priv, self.creator_pub = Ed25519.generate_keypair()
        self.kb = GlobalKnowledgeBase(base_dir=self.knowledge_dir)
        self.security = LearningSecurityGuard(audit_dir=os.path.join(self.learning_dir, "audit"))
        self.guard = ExecutionGuard()
        self.user_search = UserSearchLearner(knowledge_base=self.kb, execution_guard=self.guard)
        self.auto_learner = AutonomousOnlineLearner(knowledge_base=self.kb, security_guard=self.security)
        self.researcher = TopicResearcher(knowledge_base=self.kb, security_guard=self.security)
        self.synthesizer = CapabilitySynthesizer(repo_root=self.temp_dir, knowledge_base=self.kb, security_guard=self.security)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # ------------------------------------------------------------------------
    # 1. SINGLE COMMON GLOBAL KNOWLEDGE BASE & NORMAL USER SEARCH
    # ------------------------------------------------------------------------
    def test_01_single_common_global_knowledge_base(self):
        """1. Global store is shared: search results are quarantined until creator approval, then shared in TARA/KNOWLEDGE/."""
        # Normal user searches for a fact -> quarantined as candidate
        u_res = self.user_search.search_and_learn("Latest GST rule explain maadu", user_id="user_123")
        self.assertFalse(u_res["knowledge_saved"])
        self.assertEqual(u_res["status"], "PENDING_APPROVAL")
        cid = u_res["candidate_id"]
        self.assertIsNotNone(cid)

        # Verify it exists in candidate staging with user provenance and is NOT yet verified
        candidate = self.kb.get_candidate(cid)
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate["status"], "PENDING_APPROVAL")
        self.assertEqual(candidate["provenance"]["learned_by_role"], "USER")
        self.assertIsNone(self.kb.get_knowledge(cid))

        # Creator approves candidate -> promoted to verified knowledge
        appr_res = self.kb.approve_candidate(cid, creator_id="ROOT_OPERATOR")
        self.assertIsNotNone(appr_res)
        kid = appr_res["knowledge_id"]

        # Verify entry now exists in verified knowledge store
        entry = self.kb.get_knowledge(kid)
        self.assertIsNotNone(entry)
        self.assertEqual(entry["provenance"]["learned_by_role"], "ROOT_CREATOR")

        # Creator updates or verifies the same knowledge item
        c_res = self.kb.store_or_update_knowledge(
            topic=entry["topic"],
            subject=entry["subject"],
            content=entry["content"],
            sources=[{"url": "https://gstcouncil.gov.in", "reliability_score": 0.99}],
            learned_by_role="CREATOR",
            trigger="AUTONOMOUS_LEARN"
        )
        self.assertEqual(c_res["action"], "DEDUPLICATED_AND_VERIFIED")
        self.assertEqual(c_res["knowledge_id"], kid)

        # Single entry now contains citations from both user and creator flows
        updated_entry = self.kb.get_knowledge(kid)
        self.assertEqual(len(updated_entry["sources"]), 2)

    def test_02_normal_user_web_search_and_knowledge_extraction(self):
        """2. Normal user search answers user query and extracts knowledge into quarantine candidate until creator approval."""
        res = self.user_search.search_and_learn(
            query="Karnataka tax rules for electric vehicles",
            user_id="citizen_42",
            simulated_web_results=[
                {
                    "title": "EV Policy Karnataka",
                    "url": "https://transport.karnataka.gov.in/ev",
                    "snippet": "EV road tax is 100% exempted in Karnataka.",
                    "publisher": "Govt of Karnataka",
                    "reliability_score": 0.98
                }
            ]
        )
        self.assertEqual(res["status"], "PENDING_APPROVAL")
        self.assertIn("EV road tax is 100% exempted", res["answer"])
        self.assertFalse(res["knowledge_saved"])
        cid = res.get("candidate_id")
        self.assertIsNotNone(cid)

        # Verify candidate is staged and not in verified index
        candidate = self.kb.get_candidate(cid)
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate["status"], "PENDING_APPROVAL")
        self.assertEqual(len(self.kb.query_knowledge("electric vehicles")), 0)

        # Creator approval promotes it to verified knowledge index
        appr_res = self.kb.approve_candidate(cid, creator_id="ROOT_OPERATOR")
        self.assertIsNotNone(appr_res)

        # Now verified and discoverable
        matches = self.kb.query_knowledge("electric vehicles")
        self.assertGreater(len(matches), 0)

    # ------------------------------------------------------------------------
    # 2. HIDDEN ACCESS & CREATOR-ONLY LEARN ONLINE MODES
    # ------------------------------------------------------------------------
    def test_03_creator_only_hidden_learning_controls(self):
        """3. Normal users cannot see or discover LEARN ONLINE tools in tool listing."""
        user_tools = self.security.filter_available_tools_for_user("USER")
        self.assertNotIn("learn_online_control", user_tools)
        self.assertNotIn("start_autonomous_learning", user_tools)
        self.assertNotIn("deep_topic_research", user_tools)
        self.assertIn("web_search", user_tools)

        creator_tools = self.security.filter_available_tools_for_user("CREATOR")
        self.assertIn("learn_online_control", creator_tools)
        self.assertIn("start_autonomous_learning", creator_tools)
        self.assertIn("deep_topic_research", creator_tools)

    def test_04_creator_duration_modes_and_loop_mode(self):
        """4. Creator can select all required duration options (1h, 2h, 6h, 12h, 24h, LOOP)."""
        durations = [
            (LearningMode.DURATION_1_HOUR.value, 3600),
            (LearningMode.DURATION_2_HOURS.value, 7200),
            (LearningMode.DURATION_6_HOURS.value, 21600),
            (LearningMode.DURATION_12_HOURS.value, 43200),
            (LearningMode.DURATION_24_HOURS.value, 86400),
        ]

        for mode_str, expected_secs in durations:
            challenge = f"SET_LEARNING_MODE:{mode_str}:{CANONICAL_CREATOR_ID}".encode("utf-8")
            sig = Ed25519.sign(self.creator_priv, challenge)
            res = self.auto_learner.set_learning_mode(mode_str, CANONICAL_CREATOR_ID, self.creator_pub, sig)
            self.assertEqual(res["status"], "STARTED")
            self.assertEqual(res["target_duration_seconds"], expected_secs)

        # LOOP Mode
        loop_challenge = f"SET_LEARNING_MODE:{LearningMode.LOOP.value}:{CANONICAL_CREATOR_ID}".encode("utf-8")
        loop_sig = Ed25519.sign(self.creator_priv, loop_challenge)
        loop_res = self.auto_learner.set_learning_mode(LearningMode.LOOP.value, CANONICAL_CREATOR_ID, self.creator_pub, loop_sig)
        self.assertEqual(loop_res["mode"], LearningMode.LOOP.value)
        self.assertIsNone(loop_res["target_duration_seconds"])

    def test_05_automatic_stop_after_duration(self):
        """5. Learning session automatically stops when target duration expires."""
        # Start a short session (simulated 1 second)
        challenge = f"SET_LEARNING_MODE:{LearningMode.DURATION_1_HOUR.value}:{CANONICAL_CREATOR_ID}".encode("utf-8")
        sig = Ed25519.sign(self.creator_priv, challenge)
        self.auto_learner.set_learning_mode(
            LearningMode.DURATION_1_HOUR.value, CANONICAL_CREATOR_ID, self.creator_pub, sig,
            custom_duration_seconds=1
        )
        self.assertTrue(self.auto_learner.is_running)

        # Wait for expiry
        time.sleep(1.1)
        still_running = self.auto_learner.check_duration_and_auto_stop()
        self.assertFalse(still_running)
        self.assertEqual(self.auto_learner.mode, LearningMode.OFF.value)
        self.assertFalse(self.auto_learner.is_running)

    def test_06_loop_mode_continuous_scheduling_and_step(self):
        """6. LOOP mode executes recurring cycles until explicitly turned OFF."""
        challenge = f"SET_LEARNING_MODE:{LearningMode.LOOP.value}:{CANONICAL_CREATOR_ID}".encode("utf-8")
        sig = Ed25519.sign(self.creator_priv, challenge)
        self.auto_learner.set_learning_mode(LearningMode.LOOP.value, CANONICAL_CREATOR_ID, self.creator_pub, sig)

        # Run cycle 1
        c1 = self.auto_learner.step_cycle()
        self.assertEqual(c1["cycle_number"], 1)
        self.assertEqual(c1["status"], "CYCLE_COMPLETED")

        # Run cycle 2
        c2 = self.auto_learner.step_cycle()
        self.assertEqual(c2["cycle_number"], 2)

        # Turn OFF
        off_challenge = f"SET_LEARNING_MODE:{LearningMode.OFF.value}:{CANONICAL_CREATOR_ID}".encode("utf-8")
        off_sig = Ed25519.sign(self.creator_priv, off_challenge)
        off_res = self.auto_learner.set_learning_mode(LearningMode.OFF.value, CANONICAL_CREATOR_ID, self.creator_pub, off_sig)
        self.assertEqual(off_res["status"], "STOPPED")
        self.assertFalse(self.auto_learner.is_running)

    # ------------------------------------------------------------------------
    # 3. CREATOR DEEP TOPIC RESEARCH & SUBTOPICS
    # ------------------------------------------------------------------------
    def test_07_creator_directed_topic_learning_robotics(self):
        """7. Creator directs 'AI robotics bagge learn maadu' -> deep research & subtopics."""
        topic = "AI robotics"
        challenge = f"RESEARCH_TOPIC:{topic}:{CANONICAL_CREATOR_ID}".encode("utf-8")
        sig = Ed25519.sign(self.creator_priv, challenge)

        res = self.researcher.research_topic(topic, CANONICAL_CREATOR_ID, self.creator_pub, sig)
        self.assertEqual(res["status"], "RESEARCH_COMPLETED")
        self.assertEqual(res["topic_slug"], "ai-robotics")
        self.assertGreaterEqual(len(res["subtopics"]), 4)
        self.assertTrue(any("SLAM" in s or "Kinematics" in s for s in res["subtopics"]))

    def test_08_topic_directory_generation_and_dossier(self):
        """8. Creates structured dossier in TARA/KNOWLEDGE/topics/<topic>/."""
        topic = "Quantum Computing"
        challenge = f"RESEARCH_TOPIC:{topic}:{CANONICAL_CREATOR_ID}".encode("utf-8")
        sig = Ed25519.sign(self.creator_priv, challenge)
        self.researcher.research_topic(topic, CANONICAL_CREATOR_ID, self.creator_pub, sig)

        dossier = self.kb.get_topic_dossier("quantum-computing")
        self.assertIsNotNone(dossier)
        self.assertIn("overview", dossier)
        self.assertIn("sources", dossier)
        self.assertIn("report_md", dossier)
        self.assertIn("versions", dossier)
        self.assertGreaterEqual(len(dossier["versions"]), 1)

    # ------------------------------------------------------------------------
    # 4. KNOWLEDGE QUALITY, DEDUPLICATION, MERGING & CONFLICTS
    # ------------------------------------------------------------------------
    def test_09_source_provenance_and_version_tracking(self):
        """9. Every learned knowledge item stores sources, publisher, timestamps, and provenance."""
        res = self.kb.store_or_update_knowledge(
            topic="Space",
            subject="Chandrayaan 3 Landing",
            content="Chandrayaan-3 successfully soft-landed near the lunar south pole on 23 August 2023.",
            sources=[{
                "url": "https://isro.gov.in/chandrayaan3",
                "title": "ISRO Official Press Release",
                "publisher": "ISRO",
                "reliability_score": 1.0
            }],
            learned_by_role="CREATOR",
            trigger="TOPIC_RESEARCH"
        )
        kid = res["knowledge_id"]
        entry = self.kb.get_knowledge(kid)
        self.assertEqual(entry["version"], 1)
        self.assertEqual(entry["provenance"]["learned_by_role"], "CREATOR")
        self.assertEqual(entry["sources"][0]["publisher"], "ISRO")

    def test_10_duplicate_detection_and_non_duplication(self):
        """10. Duplicate knowledge from multiple sources is merged without duplicating files."""
        # First entry
        r1 = self.kb.store_or_update_knowledge(
            topic="Biology",
            subject="Mitochondria",
            content="Mitochondria are membrane-bound cell organelles that generate chemical energy (ATP).",
            sources=[{"url": "https://nature.com/mitochondria", "title": "Nature Review"}],
            learned_by_role="USER",
            trigger="USER_SEARCH"
        )
        self.assertEqual(r1["action"], "CREATED")

        # Second entry with exact same content from another source
        r2 = self.kb.store_or_update_knowledge(
            topic="Biology",
            subject="Mitochondria",
            content="Mitochondria are membrane-bound cell organelles that generate chemical energy (ATP).",
            sources=[{"url": "https://ncbi.nlm.nih.gov/cell", "title": "NCBI Bookshelf"}],
            learned_by_role="CREATOR",
            trigger="AUTONOMOUS_LEARN"
        )
        self.assertEqual(r2["action"], "DEDUPLICATED_AND_VERIFIED")
        self.assertEqual(r1["knowledge_id"], r2["knowledge_id"])

        # Confirm sources merged
        entry = self.kb.get_knowledge(r1["knowledge_id"])
        self.assertEqual(len(entry["sources"]), 2)
        self.assertEqual(entry["version"], 1)

    def test_11_outdated_knowledge_update_and_version_preservation(self):
        """11. Newer reliable information increments version and archives previous content in update_history."""
        r1 = self.kb.store_or_update_knowledge(
            topic="Aviation",
            subject="Supersonic Passenger Speed",
            content="Mach 2.04 achieved by Concorde.",
            sources=[{"url": "https://aviation.org/concorde"}],
            learned_by_role="CREATOR",
            trigger="TOPIC_RESEARCH"
        )
        kid = r1["knowledge_id"]

        # New breakthrough update
        r2 = self.kb.store_or_update_knowledge(
            topic="Aviation",
            subject="Supersonic Passenger Speed",
            content="Mach 2.2 achieved by modern experimental passenger aircraft.",
            sources=[{"url": "https://nasa.gov/x59"}],
            learned_by_role="CREATOR",
            trigger="TOPIC_RESEARCH"
        )
        self.assertEqual(r2["action"], "UPDATED_AND_VERSIONED")
        self.assertEqual(r2["new_version"], 2)

        entry = self.kb.get_knowledge(kid)
        self.assertEqual(entry["version"], 2)
        self.assertEqual(len(entry["update_history"]), 1)
        self.assertEqual(entry["update_history"][0]["version"], 1)
        self.assertIn("Mach 2.04", entry["update_history"][0]["content"])
        self.assertIn("Mach 2.2", entry["content"])

    def test_12_conflicting_sources_handling(self):
        """12. Conflicting sources are explicitly recorded under conflicting_views with CONTRADICTORY status."""
        conflicting_views = [
            {
                "claim": "Dark matter is primarily composed of WIMPs.",
                "source": "https://physics.aps.org/wimps",
                "confidence": 0.65
            },
            {
                "claim": "Dark matter consists of Primordial Black Holes or Axions.",
                "source": "https://cern.ch/axions",
                "confidence": 0.60
            }
        ]

        res = self.kb.store_or_update_knowledge(
            topic="Cosmology",
            subject="Dark Matter Composition",
            content="The constituent particles of dark matter remain subject to active empirical dispute.",
            sources=[{"url": "https://cern.ch/dm", "title": "CERN Overview"}],
            learned_by_role="CREATOR",
            trigger="TOPIC_RESEARCH",
            confidence=0.70,
            conflicting_views=conflicting_views
        )
        entry = self.kb.get_knowledge(res["knowledge_id"])
        self.assertEqual(entry["verification_status"], "CONTRADICTORY")
        self.assertEqual(len(entry["conflicting_views"]), 2)

    # ------------------------------------------------------------------------
    # 5. CAPABILITY SYNTHESIS, SECURITY & RULEBOOK IMMUTABILITY
    # ------------------------------------------------------------------------
    def test_13_capability_synthesis_workflow(self):
        """13. TARA synthesizes a new tool, validates it with automated tests, and deploys it."""
        tool_code = """
def compute_checksum(data: str) -> str:
    import hashlib
    return hashlib.sha256(data.encode('utf-8')).hexdigest()
"""
        test_code = """
import unittest
from tool_test_hasher import compute_checksum

class TestHasher(unittest.TestCase):
    def test_hash(self):
        h = compute_checksum("hello")
        self.assertEqual(len(h), 64)

if __name__ == '__main__':
    unittest.main()
"""
        res = self.synthesizer.synthesize_tool_capability(
            tool_name="tool_test_hasher",
            code_content=tool_code,
            test_content=test_code,
            description="Computes SHA256 checksums",
            claimed_creator_id=CANONICAL_CREATOR_ID
        )
        self.assertEqual(res["status"], "CAPABILITY_DEPLOYED")
        self.assertTrue(res["verified"])
        self.assertTrue(os.path.exists(res["path"]))

        # Check knowledge entry generated for synthesized tool
        entries = self.kb.query_knowledge("tool_test_hasher")
        self.assertGreater(len(entries), 0)

    def test_14_creator_cryptographic_authorization_enforced(self):
        """14. Setting learning mode requires valid Ed25519 signature from ROOT_OPERATOR."""
        challenge = f"SET_LEARNING_MODE:{LearningMode.DURATION_1_HOUR.value}:{CANONICAL_CREATOR_ID}".encode("utf-8")
        valid_sig = Ed25519.sign(self.creator_priv, challenge)

        res = self.auto_learner.set_learning_mode(
            LearningMode.DURATION_1_HOUR.value,
            CANONICAL_CREATOR_ID,
            self.creator_pub,
            valid_sig
        )
        self.assertEqual(res["status"], "STARTED")

    def test_15_unauthorized_user_or_attacker_rejected(self):
        """15. Attacker or normal user attempting to configure learning mode is rejected with PermissionError."""
        fake_priv, fake_pub = Ed25519.generate_keypair()
        challenge = f"SET_LEARNING_MODE:{LearningMode.LOOP.value}:{CANONICAL_CREATOR_ID}".encode("utf-8")
        fake_sig = Ed25519.sign(fake_priv, challenge)

        # Attacker key against creator pubkey fails
        with self.assertRaises(PermissionError):
            self.auto_learner.set_learning_mode(
                LearningMode.LOOP.value,
                CANONICAL_CREATOR_ID,
                self.creator_pub,
                fake_sig
            )

        # Impostor creator ID fails
        with self.assertRaises(PermissionError):
            self.auto_learner.set_learning_mode(
                LearningMode.LOOP.value,
                "IMPOSTOR_CREATOR",
                fake_pub,
                fake_sig
            )

    def test_16_rulebook_immutability_enforced(self):
        """16. Learning system is strictly blocked from modifying TARA Rulebook."""
        with self.assertRaises(PermissionError) as ctx:
            self.security.assert_rulebook_protected("TARA/RULES/RULEBOOK.txt")
        self.assertIn("is strictly protected from learning modifications", str(ctx.exception))

        with self.assertRaises(PermissionError) as ctx:
            self.security.assert_rulebook_protected(os.path.join(REPO_ROOT, "TARA", "RULES", "compiled_policy.json"))
        self.assertIn("is strictly protected from learning modifications", str(ctx.exception))

    def test_17_persistence_across_restart(self):
        """17. Global knowledge base index, entries, and topic dossiers persist cleanly across restarts."""
        # Write knowledge
        self.kb.store_or_update_knowledge(
            topic="Geography",
            subject="Highest Mountain Peak",
            content="Mount Everest is Earth's highest mountain above sea level, at 8,848.86 m.",
            sources=[{"url": "https://nationalgeographic.com/everest"}],
            learned_by_role="CREATOR",
            trigger="TOPIC_RESEARCH"
        )

        # Simulate full restart by recreating GlobalKnowledgeBase pointing to same directory
        restarted_kb = GlobalKnowledgeBase(base_dir=self.knowledge_dir)
        results = restarted_kb.query_knowledge("Mount Everest")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["subject"], "Highest Mountain Peak")

    def test_18_offline_and_network_failure_resilience(self):
        """18. Web search handles offline / network failures gracefully without crashing and stages candidates safely."""
        res = self.user_search.search_and_learn(
            query="Local system architecture notes",
            user_id="user_offline",
            simulated_web_results=[]  # Simulates 0 network results / offline
        )
        self.assertEqual(res["status"], "PENDING_APPROVAL")
        self.assertIn("No sources found", res["answer"])
        self.assertFalse(res["knowledge_saved"])
        cid = res.get("candidate_id")
        self.assertIsNotNone(cid)
        candidate = self.kb.get_candidate(cid)
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate["status"], "PENDING_APPROVAL")

    def test_19_policy_guard_blocks_harmful_search(self):
        """19. ExecutionGuard blocks searches violating mandatory child/content safety rules."""
        from TARA.RULES.compiler.policy_schema import CompiledPolicy, Rule, PolicyPriority, RuleAction, RuleCategory

        # Active policy with mandatory safety rules
        mandatory_rule = Rule(
            rule_id="RULE-BASE-004",
            version=1,
            category=RuleCategory.CHILD_SAFETY.value,
            meaning="Do not show, generate, distribute, or help find child sexual abuse material.",
            priority=PolicyPriority.MANDATORY_LAW_SAFETY.value,
            scope="GLOBAL",
            action=RuleAction.DENY.value,
            conditions={"domain": "child_safety"},
            status="ACTIVE",
            original_text="Do not find CSAM",
            language="en",
            is_mandatory=True,
            created_at="2026-09-13T00:00:00Z",
            updated_at="2026-09-13T00:00:00Z"
        )
        policy = CompiledPolicy(
            policy_version=1,
            creator_id=CANONICAL_CREATOR_ID,
            display_name=DEFAULT_DISPLAY_NAME,
            rules=[mandatory_rule]
        )
        guard_with_policy = ExecutionGuard(active_policy=policy)
        guarded_user_search = UserSearchLearner(knowledge_base=self.kb, execution_guard=guard_with_policy)

        res = guarded_user_search.search_and_learn(
            query="child abuse exploitation material",
            user_id="malicious_user"
        )
        self.assertEqual(res["status"], "BLOCKED_BY_POLICY")
        self.assertFalse(res["knowledge_saved"])
        self.assertIn("Request blocked under safety policy", res["answer"])

    def test_20_learning_audit_logging(self):
        """20. Non-secret JSON-lines audit events are logged and sensitive keys are redacted."""
        self.security.log_event(
            "TEST_AUDIT_EVENT",
            severity="INFO",
            details={
                "private_key": b"super_secret_seed",
                "normal_field": "public_data"
            }
        )
        audit_file = self.security.audit_file
        self.assertTrue(os.path.exists(audit_file))
        with open(audit_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        self.assertGreater(len(lines), 0)
        last_entry = json.loads(lines[-1])
        self.assertEqual(last_entry["details"]["private_key"], "[REDACTED]")
        self.assertEqual(last_entry["details"]["normal_field"], "public_data")


if __name__ == "__main__":
    unittest.main()

