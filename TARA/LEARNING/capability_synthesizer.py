"""
TARA/LEARNING/capability_synthesizer.py

Synthesizes new capabilities, tools, adapters, and skills during research.
Validates new code by running automated tests before deploying into TARA structure.
Strictly protects Rulebook and stores factual notes into TARA/KNOWLEDGE/.
"""

import os
import sys
import subprocess
import tempfile
from typing import Dict, List, Optional, Any

from TARA.KNOWLEDGE.knowledge_base import GlobalKnowledgeBase
from .security_guard import LearningSecurityGuard

class CapabilitySynthesizer:
    """Synthesizes, tests, and deploys validated tools or skills discovered during research."""

    def __init__(
        self,
        repo_root: Optional[str] = None,
        knowledge_base: Optional[GlobalKnowledgeBase] = None,
        security_guard: Optional[LearningSecurityGuard] = None
    ):
        if repo_root is None:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self.repo_root = repo_root
        self.kb = knowledge_base or GlobalKnowledgeBase()
        self.security = security_guard or LearningSecurityGuard()

    def synthesize_tool_capability(
        self,
        tool_name: str,
        code_content: str,
        test_content: str,
        description: str,
        claimed_creator_id: str
    ) -> Dict[str, Any]:
        """
        Creates, tests, and deploys a new tool under TARA/TOOLS/.
        Requires test execution to pass.
        """
        target_tool_path = os.path.join(self.repo_root, "TARA", "TOOLS", f"{tool_name}.py")

        # 1. Protect Rulebook from any automated mutation
        self.security.assert_rulebook_protected(target_tool_path)

        # 2. Run validation test in isolated sandbox/temp directory
        with tempfile.TemporaryDirectory(prefix="tara_cap_test_") as tmp_dir:
            test_tool_path = os.path.join(tmp_dir, f"{tool_name}.py")
            test_script_path = os.path.join(tmp_dir, f"test_{tool_name}.py")

            with open(test_tool_path, "w", encoding="utf-8") as f:
                f.write(code_content)

            with open(test_script_path, "w", encoding="utf-8") as f:
                f.write(test_content)

            proc = subprocess.run(
                [sys.executable, test_script_path],
                capture_output=True,
                text=True,
                cwd=tmp_dir
            )

            if proc.returncode != 0:
                self.security.log_event("CAPABILITY_SYNTHESIS_FAILED", severity="ERROR", details={
                    "tool": tool_name,
                    "error": proc.stderr
                })
                raise RuntimeError(f"Capability validation failed with exit code {proc.returncode}: {proc.stderr}")

        # 3. Tests passed! Deploy into production TARA/TOOLS/
        os.makedirs(os.path.dirname(target_tool_path), exist_ok=True)
        with open(target_tool_path, "w", encoding="utf-8") as f:
            f.write(code_content)

        # 4. Save knowledge regarding this capability into global TARA/KNOWLEDGE/
        self.kb.store_or_update_knowledge(
            topic="SynthesizedCapabilities",
            subject=f"Tool: {tool_name}",
            content=f"Synthesized validated tool '{tool_name}': {description}. Successfully tested.",
            sources=[{"url": f"file://{target_tool_path}", "title": f"Source code for {tool_name}", "reliability_score": 1.0}],
            learned_by_role="CREATOR",
            trigger="CAPABILITY_SYNTHESIS",
            confidence=1.0,
            verification_status="VERIFIED"
        )

        self.security.log_event("CAPABILITY_SYNTHESIS_SUCCESS", severity="INFO", details={
            "tool": tool_name,
            "path": target_tool_path
        })

        return {
            "status": "CAPABILITY_DEPLOYED",
            "tool_name": tool_name,
            "path": target_tool_path,
            "verified": True
        }
