"""
python/tara_core/experiential_bridge.py

Experiential Closed-Loop Staging Bridge for TARA Core.
Automatically translates verified operational experience from ExperientialClosedLoopLearner
into staged training samples for the dynamic dataset compiler and native self-trainer.

Lifecycle:
EXPERIENTIAL EPISODE (VERIFIED)
-> SYNTHESIZE PAIR
-> SCRUB SECRETS
-> DEDUPLICATE
-> STAGE IN SELF-LEARNING QUEUE
-> PUBLISH EVENT
-> EVALUATE CONTINUOUS LEARNING TRIGGER POLICY
"""

import os
import sys
import json
import hashlib
import logging
import threading
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tara_core.experiential_learning import ExperienceEpisode
from tara_model.dynamic_dataset_compiler import SecretScrubber
from tara_core.event_bus import TaraEventBus

logger = logging.getLogger("TARA.ExperientialBridge")


class ExperientialStagingBridge:
    """
    Connects runtime experiential learning outcomes directly into the dynamic self-training pipeline.
    """
    _instance: Optional["ExperientialStagingBridge"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self, repo_root: Optional[str] = None):
        self.repo_root = repo_root or REPO_ROOT
        self.staging_dir = os.path.join(self.repo_root, "storage", "datasets", "self_learning_staging")
        os.makedirs(self.staging_dir, exist_ok=True)
        self.staging_file = os.path.join(self.staging_dir, "staged_samples.jsonl")
        self._staged_hashes: set = set()
        self._bridge_lock = threading.RLock()
        self._load_existing_hashes()

    @classmethod
    def get_default(cls, repo_root: Optional[str] = None) -> "ExperientialStagingBridge":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(repo_root=repo_root)
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        with cls._lock:
            cls._instance = None

    def _load_existing_hashes(self) -> None:
        if os.path.exists(self.staging_file):
            try:
                with open(self.staging_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line_s = line.strip()
                        if line_s:
                            try:
                                d = json.loads(line_s)
                                prompt = d.get("prompt", "")
                                comp = d.get("completion", "")
                                h = hashlib.sha256(f"{prompt}:{comp}".encode("utf-8")).hexdigest()
                                self._staged_hashes.add(h)
                            except Exception:
                                pass
            except Exception as e:
                logger.warning(f"Error reading existing staged hashes: {e}")

    def stage_episode(self, episode: ExperienceEpisode) -> Optional[Dict[str, Any]]:
        """
        Translates a completed ExperienceEpisode into a training sample and writes it to staging.
        """
        with self._bridge_lock:
            lesson = episode.extracted_lesson
            obs = episode.observation
            if not lesson or not obs:
                return None

            prompt = f"What operational lesson did TARA learn from the task '{obs}'?"
            completion = f"Operational lesson learned: {lesson} (Outcome: {episode.outcome})."

            # Scrub secrets
            full_text = f"{prompt} {completion}"
            if SecretScrubber.contains_secret(full_text):
                logger.warning(f"Skipped staging episode '{episode.episode_id}' due to detected secrets.")
                return None

            prompt = SecretScrubber.sanitize(prompt)
            completion = SecretScrubber.sanitize(completion)

            h = hashlib.sha256(f"{prompt}:{completion}".encode("utf-8")).hexdigest()
            if h in self._staged_hashes:
                # Already staged
                return None

            self._staged_hashes.add(h)

            record = {
                "prompt": prompt,
                "completion": completion,
                "topic_family": f"experiential_{episode.episode_id[:8]}",
                "category": "experiential_learning",
                "item_name": episode.episode_id,
                "source_file": "python/tara_core/experiential_learning.py",
                "episode_id": episode.episode_id,
                "outcome": episode.outcome,
                "staged_at": datetime.now(timezone.utc).isoformat()
            }

            with open(self.staging_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            logger.info(f"Staged experiential lesson from episode '{episode.episode_id}' into self-learning queue.")

            # Publish event
            TaraEventBus.get_default().publish(
                "learning.experience_staged",
                {"episode_id": episode.episode_id, "lesson": lesson, "staged_records": 1},
                source="ExperientialStagingBridge"
            )

            return record

    def count_staged(self) -> int:
        with self._bridge_lock:
            return len(self._staged_hashes)
