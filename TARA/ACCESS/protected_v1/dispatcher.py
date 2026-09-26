"""
TARA/ACCESS/protected_v1/dispatcher.py

Conversational Command Dispatcher for capability_v1 / protected_state_v1.
Routes authenticated creator chat messages to:
- ProtectedStateManager (activation / deactivation)
- ActionBroker (source reading, modification, rollback)
- PolicyRecordStore (permanent root policies)
- StateLifecycleAdapter (existing self-destruction)
"""

import os
import re
from typing import Dict, Any, Optional, Tuple

from .manager import ProtectedStateManager, CANONICAL_CREATOR_ID
from .broker import ActionBroker
from .policy_store import PolicyRecordStore
from .destruction_adapter import StateLifecycleAdapter


class ProtectedCommandDispatcher:
    DEACTIVATION_PHRASES = {
        "deactivate capability_v1",
        "deactivate protected state",
        "deactivate protected authority",
        "lock protected state",
        "exit protected state",
        "deactivate advanced creator authority",
        "lock advanced authority",
        "close advanced authority"
    }

    def __init__(
        self,
        manager: ProtectedStateManager,
        broker: ActionBroker,
        policy_store: PolicyRecordStore,
        destruction_adapter: StateLifecycleAdapter,
        creator_auth_service: Any
    ):
        self.manager = manager
        self.broker = broker
        self.policy_store = policy_store
        self.destruction_adapter = destruction_adapter
        self.creator_auth_service = creator_auth_service

    def handle_message(
        self,
        session_token: str,
        creator_id: str,
        role: str,
        input_text: str
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluates input_text for triggers or commands.
        Returns a result dict if handled, or None if message should continue down normal cognitive loop.
        """
        if not input_text or not isinstance(input_text, str):
            return None

        clean_text = input_text.strip()
        lower_text = clean_text.lower()

        # 1. Deactivation check
        if lower_text in self.DEACTIVATION_PHRASES:
            if self.manager.is_active(session_token, self.creator_auth_service):
                ok, msg = self.manager.deactivate(session_token)
                return {
                    "handled": True,
                    "response": msg,
                    "status": "DEACTIVATED",
                    "tags": ["CREATOR-AUTHENTICATED", "CAPABILITY-V1-DEACTIVATED"]
                }

        # 2. Trigger check for activation
        if self.manager.verify_trigger(clean_text):
            if creator_id == CANONICAL_CREATOR_ID and role == "ROOT_CREATOR":
                ok, msg = self.manager.activate(session_token, clean_text, self.creator_auth_service)
                if ok:
                    return {
                        "handled": True,
                        "response": msg,
                        "status": "ACTIVATED",
                        "tags": ["CREATOR-AUTHENTICATED", "CAPABILITY-V1-ACTIVE"]
                    }
                else:
                    return {
                        "handled": True,
                        "response": msg,
                        "status": "DENIED",
                        "tags": ["CREATOR-AUTH-FAILED"]
                    }

        # 3. If protected state is NOT active, do not process privileged commands
        if not self.manager.is_active(session_token, self.creator_auth_service):
            return None

        # -------------------------------------------------------------------
        # PROTECTED STATE IS ACTIVE: Process privileged creator instructions
        # -------------------------------------------------------------------

        # A. Permanent Root Policies
        if lower_text.startswith("save permanent policy:") or lower_text.startswith("remember this permanently:") or lower_text.startswith("save this as a permanent tara policy:"):
            prefix_len = clean_text.find(":") + 1
            pol_text = clean_text[prefix_len:].strip()
            res = self.policy_store.create_policy(session_token, pol_text)
            if res.get("status") == "SUCCESS":
                pol = res["policy"]
                resp = f"✅ Permanent root policy created (ID: {pol['policy_id']}): \"{pol['policy_text']}\""
            else:
                resp = f"❌ Failed to create permanent policy: {res.get('error')}"
            return {"handled": True, "response": resp, "data": res, "tags": ["ROOT-POLICY-CREATED"]}

        if lower_text in ("list permanent policies", "show permanent policies", "get permanent policies"):
            policies = self.policy_store.list_policies(active_only=False)
            if not policies:
                resp = "ℹ️ No permanent root policies currently stored."
            else:
                items = [f"• [{p['policy_id']}] ({p['status']}) v{p['version']}: {p['policy_text']}" for p in policies]
                resp = "📋 Permanent Root Policies:\n" + "\n".join(items)
            return {"handled": True, "response": resp, "data": policies, "tags": ["ROOT-POLICIES-LISTED"]}

        if lower_text.startswith("delete permanent policy:"):
            pol_id = clean_text.split(":", 1)[1].strip()
            res = self.policy_store.delete_policy(session_token, pol_id)
            if res.get("status") == "SUCCESS":
                resp = f"✅ Permanent root policy {pol_id} deleted."
            else:
                resp = f"❌ Failed to delete policy {pol_id}: {res.get('error')}"
            return {"handled": True, "response": resp, "data": res, "tags": ["ROOT-POLICY-DELETED"]}

        # B. Source Action Commands
        if lower_text.startswith("read file:") or lower_text.startswith("read source:"):
            fpath = clean_text.split(":", 1)[1].strip()
            res = self.broker.read_source(session_token, fpath)
            if res.get("status") == "SUCCESS":
                snippet = res["content"]
                if len(snippet) > 1000:
                    snippet = snippet[:1000] + "\n... [truncated]"
                resp = f"📄 File `{res['path']}` (SHA256: `{res['sha256'][:16]}...`, {res['size_bytes']} bytes):\n```\n{snippet}\n```"
            else:
                resp = f"❌ Failed to read `{fpath}`: {res.get('error')}"
            return {"handled": True, "response": resp, "data": res, "tags": ["SOURCE-READ"]}

        if lower_text.startswith("modify file:") or lower_text.startswith("edit file:"):
            body = clean_text.split(":", 1)[1].strip()
            if "|" in body:
                fpath, content = body.split("|", 1)
                fpath = fpath.strip()
                content = content.lstrip("\r\n")
                res = self.broker.modify_source(session_token, fpath, content)
                if res.get("status") == "SUCCESS":
                    resp = f"✅ File `{res['path']}` modified successfully (change_id: `{res['change_id']}`, new SHA256: `{res['sha256'][:16]}...`)."
                else:
                    resp = f"❌ Modification failed: {res.get('error')}"
                return {"handled": True, "response": resp, "data": res, "tags": ["SOURCE-MODIFIED"]}
            else:
                fpath = body.strip()
                res = self.broker.read_source(session_token, fpath)
                if res.get("status") == "SUCCESS":
                    resp = f"File `{res['path']}` is ready for modification (current SHA256: `{res['sha256'][:16]}...`). To modify, send: `modify file: {res['path']} | <new_content>`."
                else:
                    resp = f"❌ File `{fpath}`: {res.get('error')}"
                return {"handled": True, "response": resp, "data": res, "tags": ["SOURCE-STATUS"]}

        edit_match = re.match(r"^edit\s+(?:your\s+)?([a-zA-Z0-9_/\\.-]+?)\.?$", lower_text)
        if edit_match:
            raw_target = edit_match.group(1).strip()
            if not raw_target.endswith(".py") and "." not in raw_target:
                raw_target += ".py"
            candidate_paths = [
                raw_target,
                os.path.join("python", "tara_core", raw_target),
                os.path.join("TARA", raw_target)
            ]
            found_path = None
            for cp in candidate_paths:
                full = os.path.join(self.broker.repo_root, cp)
                if os.path.isfile(full):
                    found_path = cp.replace("\\", "/")
                    break

            target_to_use = found_path or raw_target
            if os.path.isfile(os.path.join(self.broker.repo_root, target_to_use)):
                res = self.broker.read_source(session_token, target_to_use)
                resp = (
                    f"File `{target_to_use}` verified and ready for modification "
                    f"(SHA256: `{res['sha256'][:16]}...`, {res['size_bytes']} bytes). "
                    f"To apply changes with automatic rollback protection, send: `modify file: {target_to_use} | <new_content>`."
                )
                return {"handled": True, "response": resp, "data": res, "tags": ["SOURCE-INSPECTED"]}
            else:
                resp = (
                    f"Target `{target_to_use}` ready for creation/modification. "
                    f"To apply changes with automatic rollback protection, send: `modify file: {target_to_use} | <new_content>`."
                )
                return {"handled": True, "response": resp, "data": {"target": target_to_use}, "tags": ["SOURCE-READY"]}

        if lower_text.startswith("create file:") or lower_text.startswith("create source:"):
            body = clean_text.split(":", 1)[1].strip()
            if "|" in body:
                fpath, content = body.split("|", 1)
                fpath = fpath.strip()
                content = content.lstrip("\r\n")
            else:
                fpath = body.strip()
                content = ""
            res = self.broker.create_source(session_token, fpath, content)
            if res.get("status") == "SUCCESS":
                resp = f"✅ File `{res['path']}` created successfully (change_id: `{res['change_id']}`, SHA256: `{res['sha256'][:16]}...`)."
            else:
                resp = f"❌ Creation failed: {res.get('error')}"
            return {"handled": True, "response": resp, "data": res, "tags": ["SOURCE-CREATED"]}

        if lower_text.startswith("delete file:") or lower_text.startswith("delete source:"):
            fpath = clean_text.split(":", 1)[1].strip()
            res = self.broker.delete_source(session_token, fpath)
            if res.get("status") == "SUCCESS":
                resp = f"✅ File `{res['path']}` deleted successfully."
            else:
                resp = f"❌ Deletion failed: {res.get('error')}"
            return {"handled": True, "response": resp, "data": res, "tags": ["SOURCE-DELETED"]}

        if lower_text.startswith("reload component:") or lower_text.startswith("reload module:"):
            mod = clean_text.split(":", 1)[1].strip()
            res = self.broker.reload_component(session_token, mod)
            if res.get("status") == "SUCCESS":
                resp = f"✅ Module `{res['module']}` reloaded in memory."
            else:
                resp = f"❌ Reload failed: {res.get('error')}"
            return {"handled": True, "response": resp, "data": res, "tags": ["MODULE-RELOADED"]}

        # C. Lifecycle Commands
        if lower_text in ("self-destruct", "arm self destruction", "destroy tara", "arm self-destruct"):
            res = self.destruction_adapter.arm(session_token)
            if res.get("status") == "ARMED":
                resp = res.get("message")
            else:
                resp = f"❌ Could not arm self-destruction: {res.get('error')}"
            return {"handled": True, "response": resp, "data": res, "tags": ["DESTRUCTION-ARMED"]}

        if lower_text.startswith("confirm self-destruction:") or lower_text.startswith("confirm destruction:"):
            body = clean_text.split(":", 1)[1].strip()
            if "|" in body:
                arm_tok, phrase = body.split("|", 1)
                arm_tok = arm_tok.strip()
                phrase = phrase.strip()
            else:
                arm_tok = body.strip()
                phrase = "I_AUTHORIZE_COMPLETE_DESTRUCTION_OF_TARA"
            res = self.destruction_adapter.confirm_and_execute(session_token, arm_tok, phrase)
            if res.get("status") == "DESTROYED":
                resp = "💥 TARA COMPLETE IRREVERSIBLE SELF-DESTRUCTION EXECUTED."
            else:
                resp = f"❌ Self-destruction aborted: {res.get('error')}"
            return {"handled": True, "response": resp, "data": res, "tags": ["DESTRUCTION-RESULT"]}

        return None


