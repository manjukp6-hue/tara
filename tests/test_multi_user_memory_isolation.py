"""
Unit and integration tests for Multi-User Memory Isolation in TARA.
"""

import unittest
import os
import sys
import shutil

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON_DIR = os.path.join(REPO_ROOT, "python")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if PYTHON_DIR not in sys.path:
    sys.path.insert(0, PYTHON_DIR)

from tara_core.brain import TaraBrain
from TARA.MEMORY.memory_engine import MemoryEngine


class TestMultiUserMemoryIsolation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.brain = TaraBrain()
        cls.memory_engine = cls.brain.memory_engine

    def test_01_direct_memory_engine_isolation(self):
        """Verify memory_engine.query_episodes strictly isolates by actor_id."""
        # Record episodes for Alice
        ep_alice = self.memory_engine.record_episode(
            actor_id="alice_user_99",
            intent="project_discussion",
            action="chat",
            parameters={"query": "Alice's confidential project code is PROJECT_TITAN"},
            outcome="SUCCESS",
            observations={"secret": "PROJECT_TITAN"}
        )
        self.assertIsNotNone(ep_alice)

        # Record episodes for Bob
        ep_bob = self.memory_engine.record_episode(
            actor_id="bob_user_42",
            intent="vacation_planning",
            action="chat",
            parameters={"query": "Bob's secret vacation plan is HAWAII"},
            outcome="SUCCESS",
            observations={"secret": "HAWAII"}
        )
        self.assertIsNotNone(ep_bob)

        # Alice queries for secrets / projects
        alice_results = self.memory_engine.query_episodes(
            query="secret project",
            actor_id="alice_user_99",
            limit=10
        )
        for rec in alice_results:
            self.assertEqual(rec.get("actor_id"), "alice_user_99")
            self.assertNotIn("HAWAII", str(rec))

        # Bob queries for secrets / projects
        bob_results = self.memory_engine.query_episodes(
            query="secret project",
            actor_id="bob_user_42",
            limit=10
        )
        for rec in bob_results:
            self.assertEqual(rec.get("actor_id"), "bob_user_42")
            self.assertNotIn("PROJECT_TITAN", str(rec))

    def test_02_brain_pipeline_context_memory_isolation(self):
        """Verify brain.process injects only the caller's memory into retrieved context."""
        # Alice interacts with brain
        res_alice = self.brain.process(
            actor_id="alice_unique_agent",
            input_text="My secret lucky number is 777123"
        )
        self.assertEqual(res_alice["decision"], "ALLOW")

        # Bob interacts with brain
        res_bob = self.brain.process(
            actor_id="bob_unique_agent",
            input_text="My secret lucky number is 888456"
        )
        self.assertEqual(res_bob["decision"], "ALLOW")

        # Now Alice asks about lucky numbers
        alice_query_res = self.brain.process(
            actor_id="alice_unique_agent",
            input_text="What is my secret lucky number?"
        )
        retrieved_for_alice = alice_query_res.get("retrieved_context", {}).get("memory", [])
        for mem in retrieved_for_alice:
            self.assertEqual(mem.get("actor_id"), "alice_unique_agent")
            self.assertNotIn("888456", str(mem))

        # Now Bob asks about lucky numbers
        bob_query_res = self.brain.process(
            actor_id="bob_unique_agent",
            input_text="What is my secret lucky number?"
        )
        retrieved_for_bob = bob_query_res.get("retrieved_context", {}).get("memory", [])
        for mem in retrieved_for_bob:
            self.assertEqual(mem.get("actor_id"), "bob_unique_agent")
            self.assertNotIn("777123", str(mem))

    def test_03_anonymous_query_does_not_leak_user_episodes(self):
        """Verify queries for an unauthenticated / anonymous user do not return other users' private episodes."""
        anon_results = self.memory_engine.query_episodes(
            query="secret lucky number",
            actor_id="anonymous_user",
            limit=10
        )
        self.assertEqual(len(anon_results), 0)


if __name__ == "__main__":
    unittest.main()
