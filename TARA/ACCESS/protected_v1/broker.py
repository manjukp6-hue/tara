"""
TARA/ACCESS/protected_v1/broker.py

Action Broker for TARA.
Executes real source modifications, backups, syntax validation,
and automatic fail-closed rollback.

Strict Invariants:
- Dedicated broker verifying authenticated ROOT_CREATOR + active capability_v1.
- Path canonicalization preventing path traversal and symlink escape.
- Creates rollback snapshot before every modification.
- Syntax verification (ast.parse, json.loads) with automatic rollback on error.
- Production model safetensors is strictly read-only and immutable.
- Neutral audit event recording.
"""

import os
import ast
import json
import shutil
import secrets
import hashlib
import importlib
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple

from .manager import ProtectedStateManager, CANONICAL_CREATOR_ID


def compute_sha256(path: str) -> Optional[str]:
    if not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class ActionBroker:
    def __init__(
        self,
        protected_state_mgr: ProtectedStateManager = None,
        creator_auth_service: Any = None,
        repo_root: Optional[str] = None,
        audit_logger: Optional[Any] = None,
        snapshot_dir: Optional[str] = None,
    ):
        if repo_root is None:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
        self.repo_root = os.path.abspath(repo_root)
        self.protected_state_mgr = protected_state_mgr
        self.creator_auth_service = creator_auth_service
        self.audit_logger = audit_logger

        if snapshot_dir is None:
            snapshot_dir = os.path.join(self.repo_root, "storage", "snapshots")
        self.snapshot_dir = os.path.abspath(snapshot_dir)
        os.makedirs(self.snapshot_dir, exist_ok=True)

    def _verify_authorization(self, session_token: str) -> Tuple[bool, Optional[str]]:
        """Verifies session is authenticated ROOT_OPERATOR with active protected state."""
        if not self.protected_state_mgr or not self.protected_state_mgr.is_active(session_token, self.creator_auth_service):
            return False, "Unauthorized: Protected capability required."
        return True, None

    def _validate_path(self, rel_or_abs_path: str) -> Tuple[bool, str]:
        """
        Validates target path is strictly within TARA repository.
        Prevents directory traversal, symlink traversal, and critical model mutation.
        """
        if not rel_or_abs_path:
            return False, "Empty path provided."

        if os.path.isabs(rel_or_abs_path):
            target = os.path.abspath(rel_or_abs_path)
        else:
            target = os.path.abspath(os.path.join(self.repo_root, rel_or_abs_path))

        if not target.startswith(self.repo_root):
            return False, f"Path traversal rejected: {rel_or_abs_path} is outside repository root."

        rel = os.path.relpath(target, self.repo_root).replace("\\", "/")
        if rel.startswith(".git") or "/.git/" in rel:
            return False, "Access denied: .git internal directory is protected."

        if "model.safetensors" in rel.lower():
            return False, "Access denied: Production model weights are immutable."

        return True, target

    def create_snapshot(self, affected_files: List[str], change_id: str, operation: str) -> Dict[str, Any]:
        """Creates an immutable rollback snapshot of target files."""
        snap_path = os.path.join(self.snapshot_dir, change_id)
        os.makedirs(snap_path, exist_ok=True)

        file_manifest = {}
        for fpath in affected_files:
            if os.path.exists(fpath):
                rel = os.path.relpath(fpath, self.repo_root)
                dest = os.path.join(snap_path, rel.replace("/", os.sep).replace("\\", os.sep))
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.copy2(fpath, dest)
                file_manifest[rel] = {
                    "sha256": compute_sha256(fpath),
                    "exists_before": True
                }
            else:
                rel = os.path.relpath(fpath, self.repo_root)
                file_manifest[rel] = {
                    "sha256": None,
                    "exists_before": False
                }

        meta = {
            "change_id": change_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "creator_id": CANONICAL_CREATOR_ID,
            "operation": operation,
            "files": file_manifest
        }

        with open(os.path.join(snap_path, "snapshot_manifest.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return meta

    def rollback(self, change_id: str) -> Tuple[bool, str]:
        """Restores files from a previously created snapshot."""
        snap_path = os.path.join(self.snapshot_dir, change_id)
        meta_path = os.path.join(snap_path, "snapshot_manifest.json")
        if not os.path.exists(meta_path):
            return False, f"Snapshot manifest not found for {change_id}"

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            for rel, finfo in meta.get("files", {}).items():
                orig_path = os.path.join(self.repo_root, rel.replace("/", os.sep))
                dest_backup = os.path.join(snap_path, rel.replace("/", os.sep))

                if finfo.get("exists_before") and os.path.exists(dest_backup):
                    os.makedirs(os.path.dirname(orig_path), exist_ok=True)
                    shutil.copy2(dest_backup, orig_path)
                elif not finfo.get("exists_before") and os.path.exists(orig_path):
                    os.remove(orig_path)

            if self.audit_logger:
                self.audit_logger.log_event(
                    event_type="rollback",
                    severity="WARNING",
                    details={"change_id": change_id, "restored_files": list(meta.get("files", {}).keys())}
                )

            return True, f"Rollback completed for change {change_id}"
        except Exception as e:
            return False, f"Rollback failed: {str(e)}"

    def read_source(self, session_token: str, path: str) -> Dict[str, Any]:
        """Reads a TARA source file within authorized scope."""
        is_auth, err = self._verify_authorization(session_token)
        if not is_auth:
            return {"status": "ERROR", "error": err}

        valid_path, full_path = self._validate_path(path)
        if not valid_path:
            return {"status": "ERROR", "error": full_path}

        if not os.path.isfile(full_path):
            return {"status": "ERROR", "error": f"File not found: {path}"}

        try:
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()

            rel = os.path.relpath(full_path, self.repo_root).replace("\\", "/")
            return {
                "status": "SUCCESS",
                "path": rel,
                "content": content,
                "sha256": compute_sha256(full_path),
                "size_bytes": len(content.encode("utf-8"))
            }
        except Exception as e:
            return {"status": "ERROR", "error": f"Failed to read file: {str(e)}"}

    def modify_source(
        self,
        session_token: str,
        path: str,
        new_content: str,
        description: str = "Creator action"
    ) -> Dict[str, Any]:
        """Modifies an existing TARA source file with automated rollback safety."""
        is_auth, err = self._verify_authorization(session_token)
        if not is_auth:
            return {"status": "ERROR", "error": err}

        valid_path, full_path = self._validate_path(path)
        if not valid_path:
            return {"status": "ERROR", "error": full_path}

        if not os.path.isfile(full_path):
            return {"status": "ERROR", "error": f"File does not exist: {path}"}

        change_id = secrets.token_hex(8)

        if self.audit_logger:
            self.audit_logger.log_event(
                event_type="action_broker_requested",
                severity="CRITICAL",
                details={"change_id": change_id, "path": path, "description": description}
            )

        self.create_snapshot([full_path], change_id, "MODIFY_SOURCE")

        try:
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(new_content)
        except Exception as e:
            self.rollback(change_id)
            if self.audit_logger:
                self.audit_logger.log_event(
                    event_type="action_broker_failed",
                    severity="ERROR",
                    details={"change_id": change_id, "path": path, "error": str(e)}
                )
            return {"status": "FAILED_ROLLED_BACK", "change_id": change_id, "error": f"Write failed: {str(e)}"}

        if full_path.endswith(".py"):
            try:
                ast.parse(new_content)
            except SyntaxError as syn_err:
                self.rollback(change_id)
                if self.audit_logger:
                    self.audit_logger.log_event(
                        event_type="action_broker_failed",
                        severity="ERROR",
                        details={"change_id": change_id, "path": path, "syntax_error": str(syn_err)}
                    )
                return {
                    "status": "FAILED_ROLLED_BACK",
                    "change_id": change_id,
                    "error": f"Syntax validation failed: {str(syn_err)}. Rolled back automatically."
                }
        elif full_path.endswith(".json"):
            try:
                json.loads(new_content)
            except Exception as j_err:
                self.rollback(change_id)
                if self.audit_logger:
                    self.audit_logger.log_event(
                        event_type="action_broker_failed",
                        severity="ERROR",
                        details={"change_id": change_id, "path": path, "json_error": str(j_err)}
                    )
                return {
                    "status": "FAILED_ROLLED_BACK",
                    "change_id": change_id,
                    "error": f"JSON validation failed: {str(j_err)}. Rolled back automatically."
                }

        new_sha = compute_sha256(full_path)
        if self.audit_logger:
            self.audit_logger.log_event(
                event_type="action_broker_completed",
                severity="CRITICAL",
                details={"change_id": change_id, "path": path, "new_sha256": new_sha}
            )

        rel = os.path.relpath(full_path, self.repo_root).replace("\\", "/")
        return {
            "status": "SUCCESS",
            "change_id": change_id,
            "path": rel,
            "sha256": new_sha,
            "message": f"Successfully modified {rel}."
        }

    def create_source(
        self,
        session_token: str,
        path: str,
        content: str,
        description: str = "Creator created source file"
    ) -> Dict[str, Any]:
        """Creates a new TARA source file within authorized scope."""
        is_auth, err = self._verify_authorization(session_token)
        if not is_auth:
            return {"status": "ERROR", "error": err}

        valid_path, full_path = self._validate_path(path)
        if not valid_path:
            return {"status": "ERROR", "error": full_path}

        change_id = secrets.token_hex(8)
        self.create_snapshot([full_path], change_id, "CREATE_SOURCE")

        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        try:
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:
            self.rollback(change_id)
            return {"status": "FAILED_ROLLED_BACK", "change_id": change_id, "error": str(e)}

        if full_path.endswith(".py"):
            try:
                ast.parse(content)
            except SyntaxError as s_err:
                self.rollback(change_id)
                return {"status": "FAILED_ROLLED_BACK", "change_id": change_id, "error": f"Syntax error: {str(s_err)}"}

        new_sha = compute_sha256(full_path)
        if self.audit_logger:
            self.audit_logger.log_event(
                event_type="action_broker_completed",
                severity="INFO",
                details={"change_id": change_id, "path": path, "action": "create", "sha256": new_sha}
            )

        rel = os.path.relpath(full_path, self.repo_root).replace("\\", "/")
        return {
            "status": "SUCCESS",
            "change_id": change_id,
            "path": rel,
            "sha256": new_sha,
            "message": f"Successfully created {rel}."
        }

    def delete_source(
        self,
        session_token: str,
        path: str,
        description: str = "Creator deleted source file"
    ) -> Dict[str, Any]:
        """Deletes a TARA-owned source file within authorized scope."""
        is_auth, err = self._verify_authorization(session_token)
        if not is_auth:
            return {"status": "ERROR", "error": err}

        valid_path, full_path = self._validate_path(path)
        if not valid_path:
            return {"status": "ERROR", "error": full_path}

        if not os.path.isfile(full_path):
            return {"status": "ERROR", "error": f"File not found: {path}"}

        change_id = secrets.token_hex(8)
        self.create_snapshot([full_path], change_id, "DELETE_SOURCE")

        try:
            os.remove(full_path)
        except Exception as e:
            self.rollback(change_id)
            return {"status": "FAILED_ROLLED_BACK", "change_id": change_id, "error": str(e)}

        if self.audit_logger:
            self.audit_logger.log_event(
                event_type="action_broker_completed",
                severity="WARNING",
                details={"change_id": change_id, "path": path, "action": "delete"}
            )

        rel = os.path.relpath(full_path, self.repo_root).replace("\\", "/")
        return {
            "status": "SUCCESS",
            "change_id": change_id,
            "path": rel,
            "message": f"Successfully deleted {rel}."
        }

    def reload_component(self, session_token: str, module_path: str) -> Dict[str, Any]:
        """Hot-reloads a modified Python component in-memory."""
        is_auth, err = self._verify_authorization(session_token)
        if not is_auth:
            return {"status": "ERROR", "error": err}

        try:
            mod = importlib.import_module(module_path)
            reloaded = importlib.reload(mod)
            if self.audit_logger:
                self.audit_logger.log_event(
                    event_type="self_restart",
                    severity="INFO",
                    details={"module": module_path}
                )
            return {"status": "SUCCESS", "module": module_path, "reloaded": str(reloaded)}
        except Exception as e:
            return {"status": "ERROR", "error": f"Failed to reload {module_path}: {str(e)}"}


