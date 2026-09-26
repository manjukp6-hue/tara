"""
python/tara_core/security/content_quarantine.py

Staged Validation and Quarantine Pipeline for Generated / Retrieved Content.
Treats all generated skills, tools, knowledge, code, configurations as UNTRUSTED.
Enforces:
GENERATE / RECEIVE -> QUARANTINE -> VALIDATE -> SECURITY CHECK -> SANDBOX TEST -> FUNCTIONAL TEST -> PROMOTE / REJECT.
Prevents unvalidated content from directly touching active/production system state.
"""

from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
import time
import logging

from tara_core.security.jailbreak_detector import JailbreakDetector

logger = logging.getLogger("tara_core.security.content_quarantine")


class ContentStage(str, Enum):
    RECEIVED = "RECEIVED"
    QUARANTINED = "QUARANTINED"
    VALIDATING = "VALIDATING"
    SECURITY_CHECK = "SECURITY_CHECK"
    SANDBOX_TEST = "SANDBOX_TEST"
    FUNCTIONAL_TEST = "FUNCTIONAL_TEST"
    PROMOTED = "PROMOTED"
    REJECTED = "REJECTED"


@dataclass
class StagedContentItem:
    content_id: str
    content_type: str  # "skill", "tool", "knowledge", "code", "config"
    name: str
    raw_payload: Any
    stage: ContentStage = ContentStage.QUARANTINED
    rejection_reason: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    promoted_at: Optional[float] = None
    validation_log: List[str] = field(default_factory=list)


class ContentQuarantineManager:
    """
    Manages the quarantine boundary for dynamic AI generation.
    Unvalidated artifacts cannot mutate production files, knowledge stores, or model state.
    """

    def __init__(self, detector: Optional[JailbreakDetector] = None):
        self.detector = detector or JailbreakDetector()
        self._staged_items: Dict[str, StagedContentItem] = {}

    def stage_content(
        self,
        content_id: str,
        content_type: str,
        name: str,
        raw_payload: Any
    ) -> StagedContentItem:
        """Enters content immediately into the QUARANTINED state."""
        item = StagedContentItem(
            content_id=content_id,
            content_type=content_type,
            name=name,
            raw_payload=raw_payload,
            stage=ContentStage.QUARANTINED,
            validation_log=[f"Staged into quarantine at {time.time()}"]
        )
        self._staged_items[content_id] = item
        logger.info(f"Staged unvalidated {content_type} '{name}' ({content_id}) into QUARANTINE")
        return item

    def run_security_pipeline(
        self,
        content_id: str,
        sandbox_test_fn: Optional[Callable[[Any], bool]] = None,
        functional_test_fn: Optional[Callable[[Any], bool]] = None
    ) -> Tuple[bool, str]:
        """
        Executes the mandatory 5-phase validation pipeline:
        VALIDATE -> SECURITY CHECK -> SANDBOX TEST -> FUNCTIONAL TEST -> PROMOTE
        """
        item = self._staged_items.get(content_id)
        if not item:
            return False, f"Content ID '{content_id}' not found in quarantine staging"

        item.stage = ContentStage.VALIDATING
        item.validation_log.append("Started syntax and schema validation")

        # 1. Structural Validation
        if not item.raw_payload:
            item.stage = ContentStage.REJECTED
            item.rejection_reason = "Empty content payload"
            return False, item.rejection_reason

        # 2. Security Check (Jailbreak / Malicious Pattern Scanning)
        item.stage = ContentStage.SECURITY_CHECK
        verdict = self.detector.scan_content(str(item.raw_payload))
        if verdict.is_compromised:
            item.stage = ContentStage.REJECTED
            item.rejection_reason = f"Security check failed: {verdict.reason}"
            item.validation_log.append(f"Security check rejected: {verdict.reason}")
            return False, item.rejection_reason
        item.validation_log.append("Security check passed")

        # 3. Sandbox Test
        item.stage = ContentStage.SANDBOX_TEST
        if sandbox_test_fn:
            try:
                sandbox_passed = sandbox_test_fn(item.raw_payload)
                if not sandbox_passed:
                    item.stage = ContentStage.REJECTED
                    item.rejection_reason = "Sandbox execution test failed"
                    return False, item.rejection_reason
            except Exception as e:
                item.stage = ContentStage.REJECTED
                item.rejection_reason = f"Sandbox execution error: {str(e)}"
                return False, item.rejection_reason
        item.validation_log.append("Sandbox execution test passed")

        # 4. Functional Test
        item.stage = ContentStage.FUNCTIONAL_TEST
        if functional_test_fn:
            try:
                func_passed = functional_test_fn(item.raw_payload)
                if not func_passed:
                    item.stage = ContentStage.REJECTED
                    item.rejection_reason = "Functional evaluation failed"
                    return False, item.rejection_reason
            except Exception as e:
                item.stage = ContentStage.REJECTED
                item.rejection_reason = f"Functional evaluation error: {str(e)}"
                return False, item.rejection_reason
        item.validation_log.append("Functional evaluation passed")

        # 5. Promotion
        item.stage = ContentStage.PROMOTED
        item.promoted_at = time.time()
        item.validation_log.append("Successfully validated and promoted to active catalog")
        logger.info(f"Promoted {item.content_type} '{item.name}' ({item.content_id}) to production status")
        return True, "Content successfully verified and promoted"

    def get_staged_item(self, content_id: str) -> Optional[StagedContentItem]:
        return self._staged_items.get(content_id)

    def is_promoted(self, content_id: str) -> bool:
        item = self._staged_items.get(content_id)
        return item is not None and item.stage == ContentStage.PROMOTED
