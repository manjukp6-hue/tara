"""
python/tara_core/api.py

TaraCoreApi: Unified Autonomous Programming Interface for TARA Core.
Provides complete access to:
- TaraBrain (Intent parsing & orchestration)
- 17 Offline Skills Engine
- Online Learning & Research Engine
- Self-Update Engine
- Memory Engine (Episodic Trajectories & Procedural Memory)
- Creator Authority & Security Invariant Checks
"""

import os
import sys

# Ensure local imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tara_core.brain import TaraBrain
from tara_core.skills import SkillEngine
from tara_core.memory import MemoryEngine
from tara_core.online_learning import OnlineLearningEngine
from tara_core.compressor import TokenCompressor

try:
    from TARA.ACCESS.operator.operator_profile import CANONICAL_CREATOR_ID
except Exception:
    CANONICAL_CREATOR_ID = "ROOT_OPERATOR"

class TaraCoreApi:
    def __init__(self, model_checkpoint="storage/models/tara"):
        self.brain = TaraBrain(model_checkpoint=model_checkpoint)
        self.skill_engine = self.brain.skill_engine
        self.memory_engine = self.brain.memory_engine
        self.learning_engine = self.brain.learning_engine
        self.compressor = TokenCompressor()

    # ==================== TOKEN COMPRESSION ====================
    def compress_prompt(self, prompt):
        return self.compressor.compress_prompt(prompt)

    # ==================== BRAIN & ORCHESTRATION ====================
    def process(self, actor_id, input_text, context=None):
        return self.brain.process(actor_id, input_text, context)

    # ==================== SKILLS & DYNAMIC EXTENSION ====================
    def list_skills(self):
        return self.skill_engine.list_skills()

    def execute_skill(self, skill_name, params=None):
        return self.skill_engine.execute_skill(skill_name, params)

    def learn_or_update_skill(self, skill_name, code_implementation, description="", creator_id=CANONICAL_CREATOR_ID):
        return self.skill_engine.learn_or_update_skill(skill_name, code_implementation, description, creator_id)

    # ==================== ONLINE LEARNING ====================
    def learn_online(self, query, source_url=None):
        return self.learning_engine.search_and_learn(query, source_url)

    def approve_learning(self, candidate_id, creator_id=CANONICAL_CREATOR_ID):
        return self.learning_engine.approve_learning(candidate_id, creator_id)

    # ==================== SELF UPDATE ====================
    def propose_self_update(self, module_name, new_version, payload, creator_id=CANONICAL_CREATOR_ID):
        return self.learning_engine.propose_self_update(module_name, new_version, payload, creator_id)

    def rollback_update(self, update_id, reason="Creator initiated rollback"):
        return self.learning_engine.rollback_update(update_id, reason)

    # ==================== EPISODIC & PROCEDURAL MEMORY ====================
    def record_episode(self, actor_id, intent, action, parameters, outcome, observations=None, error=None):
        return self.memory_engine.record_episode(actor_id, intent, action, parameters, outcome, observations, error)

    def query_episodes(self, query=None, actor_id=None, outcome=None, limit=10):
        return self.memory_engine.query_episodes(query=query, actor_id=actor_id, outcome=outcome, limit=limit)

    def store_procedure(self, task_key, steps, description=None):
        return self.memory_engine.store_procedure(task_key, steps, description)

    def get_procedure(self, task_key):
        return self.memory_engine.get_procedure(task_key)

# Global singleton helper
_core_instance = None
def get_tara_core():
    global _core_instance
    if _core_instance is None:
        _core_instance = TaraCoreApi()
    return _core_instance
