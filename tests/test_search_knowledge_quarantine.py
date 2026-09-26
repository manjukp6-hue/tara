"""
Unit and integration tests for Live Search Knowledge Quarantine, Candidate Approval Workflow, and Session Safety.
"""

import unittest
import os
import sys
import json
import time
import threading

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON_DIR = os.path.join(REPO_ROOT, "python")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if PYTHON_DIR not in sys.path:
    sys.path.insert(0, PYTHON_DIR)

from TARA.KNOWLEDGE.knowledge_base import GlobalKnowledgeBase
from TARA.LEARNING.user_search_learner import UserSearchLearner
from tara_core.context import SessionContextManager, SessionContext


class TestSearchKnowledgeQuarantine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kb = GlobalKnowledgeBase()
        cls.learner = UserSearchLearner(knowledge_base=cls.kb)
        cls.created_candidates = []
        cls.created_entries = []

    @classmethod
    def tearDownClass(cls):
        for cid in cls.created_candidates:
            cfile = os.path.join(cls.kb.candidates_dir, f"{cid}.json")
            if os.path.exists(cfile):
                try:
                    os.remove(cfile)
                except Exception:
                    pass
        for kid in cls.created_entries:
            kfile = os.path.join(cls.kb.entries_dir, f"{kid}.json")
            if os.path.exists(kfile):
                try:
                    os.remove(kfile)
                except Exception:
                    pass
            if kid in cls.kb.index:
                del cls.kb.index[kid]
        cls.kb._save_index()

    def test_01_search_learning_is_quarantined_as_candidate(self):
        """Verify web search learner quarantines facts as PENDING_APPROVAL without auto-indexing."""
        query = "What is the official currency of Antarctica in year 2099?"
        result = self.learner.search_and_learn(query=query, user_id="normal_user_01")

        # Knowledge should NOT be saved immediately to the verified index
        self.assertFalse(result.get("knowledge_saved", True))
        self.assertEqual(result.get("status"), "PENDING_APPROVAL")
        candidate_id = result.get("candidate_id")
        self.assertIsNotNone(candidate_id)
        self.created_candidates.append(candidate_id)

        # Check candidate retrieval from KB
        candidate = self.kb.get_candidate(candidate_id)
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate.get("status"), "PENDING_APPROVAL")
        self.assertEqual(candidate.get("source_query"), query)

        # Verify knowledge_index.json does NOT list this unapproved candidate
        index_path = os.path.join(self.kb.base_dir, "knowledge_index.json")
        if os.path.exists(index_path):
            with open(index_path, "r", encoding="utf-8") as f:
                index = json.load(f)
            self.assertNotIn(candidate_id, index)

    def test_02_non_creator_cannot_approve_candidate(self):
        """Verify approval attempt by non-creator is rejected with PermissionError."""
        query = "Who won the 3020 galactic olympics?"
        result = self.learner.search_and_learn(query=query, user_id="normal_user_01")
        candidate_id = result.get("candidate_id")
        self.created_candidates.append(candidate_id)

        with self.assertRaises(PermissionError):
            self.kb.approve_candidate(candidate_id, creator_id="malicious_user")

    def test_03_creator_approval_promotes_candidate_to_verified_knowledge(self):
        """Verify creator approval successfully commits candidate to verified knowledge index."""
        query = "What is the speed of light in vacuum in km/s?"
        result = self.learner.search_and_learn(query=query, user_id="normal_user_01")
        candidate_id = result.get("candidate_id")
        self.created_candidates.append(candidate_id)

        entry_res = self.kb.approve_candidate(candidate_id, creator_id="ROOT_OPERATOR")
        self.assertIsNotNone(entry_res)
        knowledge_id = entry_res.get("knowledge_id") or candidate.get("promoted_knowledge_id")
        self.created_entries.append(knowledge_id)

        # Verify candidate status updated
        candidate = self.kb.get_candidate(candidate_id)
        self.assertEqual(candidate.get("status"), "APPROVED")
        self.assertEqual(candidate.get("approved_by"), "ROOT_OPERATOR")

        # Verify index now includes the approved entry with provenance
        with open(os.path.join(self.kb.base_dir, "knowledge_index.json"), "r", encoding="utf-8") as f:
            index = json.load(f)

        self.assertIn(knowledge_id, index)
        entry = index[knowledge_id]
        self.assertEqual(entry.get("status"), "VERIFIED")

    def test_04_session_context_ttl_expiration_and_cleanup(self):
        """Verify SessionContextManager expires stale sessions after TTL and preserves active ones."""
        manager = SessionContextManager(ttl_seconds=0.4, max_sessions=10)

        # Create session 1 and record a turn
        s1 = manager.get_or_create("sess_old", "user_old")
        s1.record_turn("hello", "hi", slots={"key": "val"})
        self.assertEqual(len(s1.history), 1)

        # Sleep past TTL
        time.sleep(0.5)

        # Trigger cleanup
        expired_count = manager.cleanup_expired_sessions()
        self.assertGreaterEqual(expired_count, 1)

        # Getting old session creates fresh one with empty history
        s1_new = manager.get_or_create("sess_old", "user_old")
        self.assertEqual(len(s1_new.history), 0)

    def test_05_session_context_thread_safety(self):
        """Verify SessionContextManager and SessionContext handle concurrent reads/writes safely."""
        manager = SessionContextManager(ttl_seconds=60.0, max_turns_per_session=200, max_sessions=100)
        session = manager.get_or_create("concurrent_sess", "worker_user")

        def worker(w_id):
            for i in range(20):
                session.record_turn(
                    user_input=f"worker_{w_id}_input_{i}",
                    response=f"worker_{w_id}_resp_{i}",
                    slots={"idx": i}
                )
                manager.get_or_create(f"worker_sess_{w_id}_{i}", f"worker_{w_id}")

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Check that 100 turns were recorded safely without corruption
        self.assertEqual(len(session.history), 100)


if __name__ == "__main__":
    unittest.main()
