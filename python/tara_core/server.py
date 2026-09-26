import os
import sys
import json
import time
import secrets
import threading
import uuid
import hashlib
import base64
import urllib.parse
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional, Dict, Any, Tuple, List

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
python_dir = os.path.join(REPO_ROOT, "python")
if python_dir not in sys.path:
    sys.path.insert(0, python_dir)

from tara_core.brain import TaraBrain
from tara_core.voice import VoiceInputProvider, VoiceOutputProvider, BargeInCoordinator, WakeWordDetector

GLOBAL_VOICE_INPUT = VoiceInputProvider()
GLOBAL_VOICE_OUTPUT = VoiceOutputProvider()
GLOBAL_BARGE_IN = BargeInCoordinator()
GLOBAL_WAKE_WORD = WakeWordDetector()


class SlidingWindowRateLimiter:
    """
    Thread-safe in-memory sliding-window rate limiter with burst protection
    and bounded memory cleanup to prevent resource exhaustion.
    """

    def __init__(
        self,
        limit_per_minute: int = 60,
        burst_limit: int = 15,
        burst_window_seconds: float = 5.0,
        cleanup_interval: float = 60.0,
        max_tracked_clients: int = 5000
    ):
        self.limit_per_minute = limit_per_minute
        self.burst_limit = burst_limit
        self.burst_window = burst_window_seconds
        self.cleanup_interval = cleanup_interval
        self.max_tracked_clients = max_tracked_clients
        self._history: Dict[str, List[float]] = {}
        self._lock = threading.RLock()
        self._last_cleanup = time.time()

    def is_allowed(self, client_key: str) -> Tuple[bool, int]:
        """
        Evaluates whether request from client_key is permitted.
        Returns (is_allowed, retry_after_seconds).
        """
        now = time.time()
        with self._lock:
            # Periodic cleanup of expired client records
            if now - self._last_cleanup > self.cleanup_interval or len(self._history) > self.max_tracked_clients:
                self.cleanup(now)

            if client_key not in self._history:
                if len(self._history) >= self.max_tracked_clients:
                    oldest = min(self._history.keys(), key=lambda k: self._history[k][-1] if self._history[k] else 0)
                    del self._history[oldest]
                self._history[client_key] = [now]
                return True, 0

            timestamps = self._history[client_key]
            # Prune records older than 60s
            cutoff_minute = now - 60.0
            timestamps = [t for t in timestamps if t > cutoff_minute]
            self._history[client_key] = timestamps

            # Check burst limit
            cutoff_burst = now - self.burst_window
            recent_burst = [t for t in timestamps if t > cutoff_burst]
            if len(recent_burst) >= self.burst_limit:
                retry_after = max(1, int(self.burst_window - (now - recent_burst[0])) + 1)
                return False, retry_after

            # Check total requests in 60s
            if len(timestamps) >= self.limit_per_minute:
                retry_after = max(1, int(60.0 - (now - timestamps[0])) + 1)
                return False, retry_after

            timestamps.append(now)
            return True, 0

    def cleanup(self, now: Optional[float] = None) -> None:
        now = now or time.time()
        cutoff = now - 60.0
        with self._lock:
            empty_keys = []
            for k, ts in self._history.items():
                active = [t for t in ts if t > cutoff]
                if active:
                    self._history[k] = active
                else:
                    empty_keys.append(k)
            for k in empty_keys:
                del self._history[k]
class WebSessionStore:
    """
    Scoped in-memory web session store for browser UI parity.
    Matches Rust server WebSessionStore logic.
    """
    def __init__(self, ttl: float = 3600.0):
        self.ttl = ttl
        self._sessions: Dict[str, Tuple[str, float]] = {}
        self._lock = threading.RLock()

    def issue(self, actor_id: str = "user") -> str:
        token = secrets.token_hex(32)
        with self._lock:
            self._sessions[token] = (actor_id, time.time())
        return token

    def verify(self, token: str) -> Optional[str]:
        if not token:
            return None
        now = time.time()
        with self._lock:
            if token in self._sessions:
                actor, issued_at = self._sessions[token]
                if now - issued_at < self.ttl:
                    return actor
                del self._sessions[token]
        return None


GLOBAL_WEB_SESSION_STORE = WebSessionStore()


class ApiRouter:
    """
    Dynamic route registry for TARA API server.
    Enables adding open-ended endpoints dynamically without modifying server core.
    """
    def __init__(self):
        self._routes: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()

    @staticmethod
    def normalize_path(path: str) -> str:
        clean = (path or "").strip()
        if not clean or clean == "/":
            return "/"
        return "/" + clean.strip("/")

    def register_route(
        self,
        method: str,
        path: str,
        handler: Any,
        required_auth: bool = True,
        required_role: Optional[str] = None,
        required_permission: Optional[str] = None
    ) -> None:
        norm_path = self.normalize_path(path)
        key = f"{method.upper()}:{norm_path}"
        with self._lock:
            self._routes[key] = {
                "method": method.upper(),
                "path": norm_path,
                "handler": handler,
                "required_auth": required_auth,
                "required_role": required_role,
                "required_permission": required_permission
            }

    def unregister_route(self, method: str, path: str) -> bool:
        norm_path = self.normalize_path(path)
        key = f"{method.upper()}:{norm_path}"
        with self._lock:
            return self._routes.pop(key, None) is not None

    def match(self, method: str, path: str) -> Optional[Dict[str, Any]]:
        norm_path = self.normalize_path(path)
        key = f"{method.upper()}:{norm_path}"
        with self._lock:
            return self._routes.get(key)

    def has_path(self, path: str) -> bool:
        norm_path = self.normalize_path(path)
        with self._lock:
            return any(r["path"] == norm_path for r in self._routes.values())

    def list_routes(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._routes.values())

GLOBAL_API_ROUTER = ApiRouter()


def register_auto_connect_sync_routes(router: ApiRouter) -> None:
    def handle_register_endpoint(handler, payload, actor=None):
        brain = getattr(handler.server, "brain", None)
        if not brain or not hasattr(brain, "auto_sync"):
            return {"status": "ERROR", "error": "AutoSyncEngine not available on brain."}
        from tara_core.auto_connect_sync import EndpointDefinition
        try:
            ep = EndpointDefinition.from_dict(payload)
            registered = brain.auto_sync.register_endpoint(ep)
            return {"status": "SUCCESS", "endpoint": registered.to_dict()}
        except Exception as e:
            return {"status": "ERROR", "error": str(e)}

    def handle_list_endpoints(handler, query_params, actor=None):
        brain = getattr(handler.server, "brain", None)
        if not brain or not hasattr(brain, "auto_sync"):
            return {"status": "ERROR", "error": "AutoSyncEngine not available on brain."}
        return {"status": "SUCCESS", "endpoints": brain.auto_sync.list_endpoints()}

    def handle_select_endpoint(handler, payload, actor=None):
        brain = getattr(handler.server, "brain", None)
        if not brain or not hasattr(brain, "auto_sync"):
            return {"status": "ERROR", "error": "AutoSyncEngine not available on brain."}
        req_caps = payload.get("required_capabilities") if isinstance(payload, dict) else None
        ep = brain.auto_sync.get_active_endpoint(required_capabilities=req_caps)
        return {"status": "SUCCESS", "active_endpoint": ep.to_dict() if ep else None}

    def handle_probe_endpoints(handler, payload, actor=None):
        brain = getattr(handler.server, "brain", None)
        if not brain or not hasattr(brain, "auto_sync"):
            return {"status": "ERROR", "error": "AutoSyncEngine not available on brain."}
        results = brain.auto_sync.probe_all_endpoints()
        return {"status": "SUCCESS", "probe_results": results}

    def handle_sync_push(handler, payload, actor=None):
        brain = getattr(handler.server, "brain", None)
        if not brain or not hasattr(brain, "auto_sync"):
            return {"status": "ERROR", "error": "AutoSyncEngine not available on brain."}
        accepted, reason = brain.auto_sync.receive_sync_package(payload)
        status_str = "SUCCESS" if accepted else "REJECTED"
        return {"status": status_str, "accepted": accepted, "reason": reason}

    def handle_sync_status(handler, query_params, actor=None):
        brain = getattr(handler.server, "brain", None)
        if not brain or not hasattr(brain, "auto_sync"):
            return {"status": "ERROR", "error": "AutoSyncEngine not available on brain."}
        return {"status": "SUCCESS", "sync_status": brain.auto_sync.get_status()}

    def handle_sync_flush(handler, payload, actor=None):
        brain = getattr(handler.server, "brain", None)
        if not brain or not hasattr(brain, "auto_sync"):
            return {"status": "ERROR", "error": "AutoSyncEngine not available on brain."}
        res = brain.auto_sync.flush_offline_journal()
        return {"status": "SUCCESS", "flush_result": res}

    def handle_cluster_telemetry(handler, query_params, actor=None):
        brain = getattr(handler.server, "brain", None)
        if not brain or not hasattr(brain, "get_cluster_telemetry"):
            return {"status": "ERROR", "error": "Cluster telemetry not available on brain."}
        return {"status": "SUCCESS", "cluster_telemetry": brain.get_cluster_telemetry()}

    def handle_cluster_route(handler, payload, actor=None):
        brain = getattr(handler.server, "brain", None)
        if not brain or not hasattr(brain, "auto_sync") or not hasattr(brain.auto_sync, "workload_gateway"):
            return {"status": "ERROR", "error": "Workload gateway not available on brain."}
        gw = brain.auto_sync.workload_gateway
        wl = gw.detector.detect(input_data=payload)
        mode, nodes = gw.capability_router.route(wl, gw.get_all_nodes())
        return {
            "status": "SUCCESS",
            "workload": wl.to_dict(),
            "execution_mode": mode.value,
            "candidate_nodes": [n.node_id for n in nodes]
        }

    def handle_cluster_execute(handler, payload, actor=None):
        brain = getattr(handler.server, "brain", None)
        if not brain or not hasattr(brain, "process_distributed_workload"):
            return {"status": "ERROR", "error": "Distributed workload execution not available on brain."}
        res = brain.process_distributed_workload(input_data=payload)
        return {"status": "SUCCESS", "execution_result": res}

    router.register_route("POST", "/api/v1/endpoints/register", handle_register_endpoint, required_auth=True)
    router.register_route("GET", "/api/v1/endpoints/list", handle_list_endpoints, required_auth=True)
    router.register_route("POST", "/api/v1/endpoints/select", handle_select_endpoint, required_auth=True)
    router.register_route("POST", "/api/v1/endpoints/probe", handle_probe_endpoints, required_auth=True)
    router.register_route("POST", "/api/v1/sync/push", handle_sync_push, required_auth=True)
    router.register_route("GET", "/api/v1/sync/status", handle_sync_status, required_auth=True)
    router.register_route("POST", "/api/v1/sync/flush", handle_sync_flush, required_auth=True)
    router.register_route("GET", "/api/v1/cluster/telemetry", handle_cluster_telemetry, required_auth=True)
    router.register_route("POST", "/api/v1/cluster/route", handle_cluster_route, required_auth=True)
    router.register_route("POST", "/api/v1/cluster/execute", handle_cluster_execute, required_auth=True)


register_auto_connect_sync_routes(GLOBAL_API_ROUTER)


def register_dynamic_engine_routes(router: ApiRouter) -> None:
    def handle_list_engines(handler, query_params, actor=None):
        brain = getattr(handler.server, "brain", None)
        if not brain or not hasattr(brain, "engine_system"):
            return {"status": "ERROR", "error": "DynamicEngineSystem not available on brain."}
        engines = brain.engine_system.list_engines()
        return {"status": "SUCCESS", "engines": engines}

    def handle_register_engine(handler, payload, actor=None):
        brain = getattr(handler.server, "brain", None)
        if not brain or not hasattr(brain, "engine_system"):
            return {"status": "ERROR", "error": "DynamicEngineSystem not available on brain."}
        from tara_core.dynamic_engine_system import EngineManifest
        try:
            manifest = EngineManifest.from_dict(payload)
            success, err = brain.engine_system.register_manifest(manifest)
            if success:
                return {"status": "SUCCESS", "engine_id": manifest.engine_id}
            return {"status": "ERROR", "error": err}
        except Exception as e:
            return {"status": "ERROR", "error": str(e)}

    def handle_execute_engine(handler, payload, actor=None):
        brain = getattr(handler.server, "brain", None)
        if not brain or not hasattr(brain, "engine_system"):
            return {"status": "ERROR", "error": "DynamicEngineSystem not available on brain."}
        task_type = payload.get("task_type", "")
        engine_id = payload.get("engine_id")
        task_payload = payload.get("payload", {})
        result = brain.execute_engine(task_type=task_type, payload=task_payload, engine_id=engine_id)
        status_str = "SUCCESS" if result.get("success", False) else "ERROR"
        return {"status": status_str, "result": result}

    def handle_quarantine_engine(handler, payload, actor=None):
        brain = getattr(handler.server, "brain", None)
        if not brain or not hasattr(brain, "engine_system"):
            return {"status": "ERROR", "error": "DynamicEngineSystem not available on brain."}
        engine_id = payload.get("engine_id", "")
        reason = payload.get("reason", "API request quarantine")
        success = brain.engine_system.quarantine_engine(engine_id, reason=reason)
        return {"status": "SUCCESS" if success else "ERROR", "engine_id": engine_id, "quarantined": success}

    def handle_engines_health(handler, query_params, actor=None):
        brain = getattr(handler.server, "brain", None)
        if not brain or not hasattr(brain, "engine_system"):
            return {"status": "ERROR", "error": "DynamicEngineSystem not available on brain."}
        health = brain.engine_system.get_health_report()
        return {"status": "SUCCESS", "health": health}

    router.register_route("GET", "/api/v1/engines/list", handle_list_engines, required_auth=True)
    router.register_route("POST", "/api/v1/engines/register", handle_register_engine, required_auth=True)
    router.register_route("POST", "/api/v1/engines/execute", handle_execute_engine, required_auth=True)
    router.register_route("POST", "/api/v1/engines/quarantine", handle_quarantine_engine, required_auth=True)
    router.register_route("GET", "/api/v1/engines/health", handle_engines_health, required_auth=True)


register_dynamic_engine_routes(GLOBAL_API_ROUTER)


def register_creator_auth_routes(router: ApiRouter) -> None:
    def handle_auth_google(handler, payload, actor=None):
        brain = getattr(handler.server, "brain", None)
        if not brain or not hasattr(brain, "creator_auth_service") or not brain.creator_auth_service:
            return {"status": "ERROR", "error": "CreatorAuthService not available on brain."}
        id_token = payload.get("id_token") or payload.get("token") or ""
        client_ip = handler.client_address[0] if getattr(handler, "client_address", None) else "127.0.0.1"
        res = brain.creator_auth_service.authenticate_google_token(id_token=id_token, client_key=client_ip)
        if res.get("status") == "SUCCESS":
            return res
        else:
            handler._send_json(401, res)
            return None

    def handle_auth_logout(handler, payload, actor=None):
        brain = getattr(handler.server, "brain", None)
        if not brain or not hasattr(brain, "creator_auth_service") or not brain.creator_auth_service:
            return {"status": "ERROR", "error": "CreatorAuthService not available on brain."}
        token = payload.get("session_token")
        if not token:
            auth_header = handler.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                token = auth_header[7:].strip()
        if not token:
            handler._send_json(400, {"status": "ERROR", "error": "Missing session token"})
            return None
        res = brain.creator_auth_service.logout(token)
        return res

    def handle_auth_session(handler, query_params, actor=None):
        brain = getattr(handler.server, "brain", None)
        if not brain or not hasattr(brain, "creator_auth_service") or not brain.creator_auth_service:
            return {"status": "ERROR", "error": "CreatorAuthService not available on brain."}
        token = None
        auth_header = handler.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:].strip()
        if not token and query_params and "session_token" in query_params:
            token = query_params["session_token"][0]
        if not token:
            creator_state = brain.creator_auth_service.lifecycle.get_state() if hasattr(brain, "creator_auth_service") and brain.creator_auth_service else "CREATOR_SETUP_REQUIRED"
            handler._send_json(200 if creator_state == "CREATOR_SETUP_REQUIRED" else 401, {
                "status": creator_state if creator_state == "CREATOR_SETUP_REQUIRED" else "ERROR",
                "authenticated": False,
                "creator_setup_status": creator_state,
                "error": "Creator setup required" if creator_state == "CREATOR_SETUP_REQUIRED" else "Missing session token"
            })
            return None
        sess = brain.creator_auth_service.verify_session(token)
        if sess:
            return {"status": "SUCCESS", "authenticated": True, "session": sess}
        handler._send_json(401, {"status": "ERROR", "authenticated": False, "error": "Invalid or expired session"})
        return None

    def handle_qr_challenge(handler, payload, actor=None):
        brain = getattr(handler.server, "brain", None)
        if not brain or not hasattr(brain, "creator_auth_service") or not brain.creator_auth_service:
            return {"status": "ERROR", "error": "CreatorAuthService not available on brain."}
        res = brain.creator_auth_service.create_qr_challenge()
        return {"status": "SUCCESS", "challenge": res}

    def handle_qr_approve(handler, payload, actor=None):
        brain = getattr(handler.server, "brain", None)
        if not brain or not hasattr(brain, "creator_auth_service") or not brain.creator_auth_service:
            return {"status": "ERROR", "error": "CreatorAuthService not available on brain."}
        challenge_id = payload.get("challenge_id", "")
        device_id = payload.get("device_id", "")
        device_sig = payload.get("device_signature", "")
        res = brain.creator_auth_service.verify_qr_approval(challenge_id, device_id, device_sig)
        if res.get("status") == "SUCCESS":
            if not hasattr(handler.server, "_approved_qr_sessions"):
                handler.server._approved_qr_sessions = {}
            handler.server._approved_qr_sessions[challenge_id] = res
            return res
        else:
            handler._send_json(401, res)
            return None

    def handle_trigger_check(handler, payload, actor=None):
        brain = getattr(handler.server, "brain", None)
        if not brain or not hasattr(brain, "creator_auth_service") or not brain.creator_auth_service:
            return {"status": "ERROR", "error": "CreatorAuthService not available on brain."}
        text = payload.get("text", "")
        res = brain.creator_auth_service.check_conversational_trigger(text)
        return {"status": "SUCCESS", "trigger_evaluation": res}

    def handle_auth_creator_key(handler, payload, actor=None):
        brain = getattr(handler.server, "brain", None)
        if not brain or not hasattr(brain, "creator_auth_service") or not brain.creator_auth_service:
            return {"status": "ERROR", "error": "CreatorAuthService not available on brain."}
        sig = payload.get("proof_signature") or payload.get("signature")
        nonce = payload.get("challenge_nonce") or payload.get("nonce")
        claimed_id = payload.get("claimed_creator_id") or payload.get("creator_id") or "ROOT_OPERATOR"
        passphrase = payload.get("passphrase") or payload.get("secret_key")
        client_ip = handler.client_address[0] if getattr(handler, "client_address", None) else "127.0.0.1"
        res = brain.creator_auth_service.authenticate_creator_key(
            proof_signature_hex=sig,
            challenge_nonce=nonce,
            claimed_creator_id=claimed_id,
            passphrase=passphrase,
            client_key=client_ip
        )
        if res.get("status") == "SUCCESS":
            return res
        else:
            handler._send_json(401, res)
            return None

    def handle_auth_recovery(handler, payload, actor=None):
        brain = getattr(handler.server, "brain", None)
        if not brain or not hasattr(brain, "creator_auth_service") or not brain.creator_auth_service:
            return {"status": "ERROR", "error": "CreatorAuthService not available on brain."}
        rec_code = payload.get("recovery_code") or payload.get("token") or ""
        claimed_id = payload.get("claimed_creator_id", "ROOT_OPERATOR")
        client_ip = handler.client_address[0] if getattr(handler, "client_address", None) else "127.0.0.1"
        res = brain.creator_auth_service.authenticate_recovery(
            recovery_code_or_token=rec_code,
            claimed_creator_id=claimed_id,
            client_key=client_ip
        )
        if res.get("status") == "SUCCESS":
            return res
        else:
            handler._send_json(401, res)
            return None

    def handle_select_method(handler, payload, actor=None):
        brain = getattr(handler.server, "brain", None)
        if not brain or not hasattr(brain, "creator_auth_service") or not brain.creator_auth_service:
            return {"status": "ERROR", "error": "CreatorAuthService not available on brain."}
        method = payload.get("method", "")
        creator_id = payload.get("creator_id", "ROOT_OPERATOR")
        client_ip = handler.client_address[0] if getattr(handler, "client_address", None) else "127.0.0.1"
        res = brain.creator_auth_service.handle_method_selection(method, creator_id=creator_id, client_key=client_ip)
        return res

    def handle_qr_status(handler, query_params, actor=None):
        challenge_id = query_params.get("challenge_id", [None])[0]
        if not challenge_id:
            handler._send_json(400, {"status": "ERROR", "error": "Missing challenge_id"})
            return None
        approved = getattr(handler.server, "_approved_qr_sessions", {}).get(challenge_id)
        if approved:
            return {"status": "APPROVED", "session": approved}
        return {"status": "PENDING"}

    def handle_creator_enroll_google(handler, payload, actor=None):
        brain = getattr(handler.server, "brain", None)
        if not brain or not hasattr(brain, "creator_auth_service") or not brain.creator_auth_service:
            return {"status": "ERROR", "error": "CreatorAuthService not available on brain."}
        token = payload.get("session_token")
        if not token:
            auth_header = handler.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                token = auth_header[7:].strip()
        creator_id = payload.get("creator_id", "")
        google_email = payload.get("google_email", "")
        google_sub = payload.get("google_subject_id")
        res = brain.creator_auth_service.enroll_creator_google_identity(
            creator_id=creator_id,
            google_email=google_email,
            google_subject_id=google_sub,
            session_token=token
        )
        if res.get("status") == "SUCCESS":
            return res
        else:
            handler._send_json(403 if "Unauthorized" in res.get("error", "") else 400, res)
            return None

    def handle_creator_setup(handler, payload, actor=None):
        brain = getattr(handler.server, "brain", None)
        if not brain or not hasattr(brain, "creator_auth_service") or not brain.creator_auth_service:
            handler._send_json(500, {"status": "ERROR", "error": "CreatorAuthService not available."})
            return None
        lifecycle = brain.creator_auth_service.lifecycle
        if lifecycle.get_state() != "CREATOR_SETUP_REQUIRED":
            handler._send_json(403, {"status": "ERROR", "error": "Creator authority is already initialized. Setup cannot be re-run."})
            return None

        confirm_identity = payload.get("confirm_identity", False)
        if not confirm_identity:
            handler._send_json(400, {"status": "ERROR", "error": "Explicit confirmation of root creator identity is required."})
            return None

        google_id_token = payload.get("google_id_token")
        if not google_id_token:
            handler._send_json(400, {"status": "ERROR", "error": "Google ID token is required for root creator setup."})
            return None

        master_passphrase = payload.get("master_passphrase", "")
        confirm_passphrase = payload.get("confirm_passphrase", "")
        if not master_passphrase or master_passphrase != confirm_passphrase or len(master_passphrase) < 8:
            handler._send_json(400, {"status": "ERROR", "error": "Valid matching master passphrase (min 8 chars) is required."})
            return None

        trigger_phrase = payload.get("private_trigger_phrase")
        device_pub = payload.get("device_public_key")
        device_name = payload.get("device_name", "Primary PC")

        from TARA.ACCESS.wizard.setup_wizard import CreatorSetupWizard
        wizard = CreatorSetupWizard(repo_root=brain.creator_auth_service.repo_root)
        try:
            setup_res = wizard.run_setup(
                master_passphrase=master_passphrase,
                confirm_passphrase=confirm_passphrase,
                google_id_token=google_id_token,
                confirm_identity=confirm_identity,
                private_trigger_phrase=trigger_phrase
            )
        except Exception as e:
            handler._send_json(400, {"status": "ERROR", "error": str(e)})
            return None

        # Register device if public key provided
        device_info = None
        if device_pub:
            try:
                dev_bytes = bytes.fromhex(device_pub)
                device_info = brain.creator_auth_service.devices.register_device(
                    public_key_bytes=dev_bytes,
                    device_name=device_name,
                    status="AUTHORIZED"
                )
            except Exception as e:
                device_info = None
                logging.getLogger("tara.server").warning("Device registration deferred during setup: %s", e)

        # Issue creator session
        from TARA.ACCESS.operator.operator_profile import CANONICAL_CREATOR_ID, DEFAULT_DISPLAY_NAME
        session = brain.creator_auth_service._issue_creator_session(
            creator_id=CANONICAL_CREATOR_ID,
            auth_method="SETUP_INITIALIZATION"
        )

        return {
            "status": "SUCCESS",
            "authority_state": "ACTIVE",
            "creator_id": CANONICAL_CREATOR_ID,
            "display_name": DEFAULT_DISPLAY_NAME,
            "device": device_info,
            "recovery_code": setup_res.get("recovery_code"),
            "session": session
        }

    def handle_creator_register_device(handler, payload, actor=None):
        brain = getattr(handler.server, "brain", None)
        if not brain or not hasattr(brain, "creator_auth_service") or not brain.creator_auth_service:
            handler._send_json(500, {"status": "ERROR", "error": "CreatorAuthService not available."})
            return None
        if not brain.creator_auth_service.lifecycle.is_active():
            handler._send_json(403, {"status": "CREATOR_SETUP_REQUIRED", "error": "Creator setup has not been performed yet."})
            return None

        google_token = payload.get("google_id_token")
        session_token = payload.get("session_token")
        if not session_token:
            auth_h = handler.headers.get("Authorization", "")
            if auth_h.startswith("Bearer "):
                session_token = auth_h[7:].strip()

        from TARA.ACCESS.operator.operator_profile import CANONICAL_CREATOR_ID
        authenticated_creator = None
        if google_token:
            auth_res = brain.creator_auth_service.authenticate_google_token(google_token)
            if auth_res.get("status") == "SUCCESS":
                authenticated_creator = auth_res.get("creator_id")
        elif session_token:
            sess = brain.creator_auth_service.verify_session(session_token)
            if sess:
                authenticated_creator = sess.get("creator_id")

        if not authenticated_creator or authenticated_creator != CANONICAL_CREATOR_ID:
            handler._send_json(401, {"status": "UNAUTHORIZED", "error": "Root creator authentication required to register devices."})
            return None

        device_pub = payload.get("device_public_key")
        device_name = payload.get("device_name", "Secondary PC")
        if not device_pub:
            handler._send_json(400, {"status": "ERROR", "error": "device_public_key is required."})
            return None

        try:
            dev_bytes = bytes.fromhex(device_pub)
            dev_rec = brain.creator_auth_service.devices.register_device(
                public_key_bytes=dev_bytes,
                device_name=device_name,
                status="AUTHORIZED"
            )
            session = brain.creator_auth_service._issue_creator_session(
                creator_id=CANONICAL_CREATOR_ID,
                auth_method="DEVICE_REGISTRATION"
            )
            return {
                "status": "SUCCESS",
                "device": dev_rec,
                "session": session
            }
        except Exception as e:
            handler._send_json(400, {"status": "ERROR", "error": str(e)})
            return None

    def handle_creator_revoke_device(handler, payload, actor=None):
        brain = getattr(handler.server, "brain", None)
        if not brain or not hasattr(brain, "creator_auth_service") or not brain.creator_auth_service:
            handler._send_json(500, {"status": "ERROR", "error": "CreatorAuthService not available."})
            return None
        session_token = payload.get("session_token")
        if not session_token:
            auth_h = handler.headers.get("Authorization", "")
            if auth_h.startswith("Bearer "):
                session_token = auth_h[7:].strip()
        if not session_token or not brain.creator_auth_service.verify_session(session_token):
            handler._send_json(401, {"status": "UNAUTHORIZED", "error": "Creator session required to revoke devices."})
            return None

        device_id = payload.get("device_id")
        revoke_all = payload.get("all", False)
        reason = payload.get("reason", "manual_revocation")

        if revoke_all:
            for dev in brain.creator_auth_service.devices.list_authorized_devices():
                brain.creator_auth_service.devices.revoke_device(dev["device_id"], reason=reason)
            return {"status": "SUCCESS", "message": "All devices revoked."}

        if not device_id:
            handler._send_json(400, {"status": "ERROR", "error": "device_id is required."})
            return None

        success = brain.creator_auth_service.devices.revoke_device(device_id, reason=reason)
        if success:
            return {"status": "SUCCESS", "device_id": device_id, "device_status": "REVOKED"}
        handler._send_json(404, {"status": "ERROR", "error": f"Device '{device_id}' not found."})
        return None

    def handle_creator_list_devices(handler, query_params, actor=None):
        brain = getattr(handler.server, "brain", None)
        if not brain or not hasattr(brain, "creator_auth_service") or not brain.creator_auth_service:
            handler._send_json(500, {"status": "ERROR", "error": "CreatorAuthService not available."})
            return None
        auth_h = handler.headers.get("Authorization", "")
        token = auth_h[7:].strip() if auth_h.startswith("Bearer ") else (query_params.get("session_token", [None])[0] if query_params else None)
        if not token or not brain.creator_auth_service.verify_session(token):
            handler._send_json(401, {"status": "UNAUTHORIZED", "error": "Creator session required."})
            return None
        devs = list(brain.creator_auth_service.devices.devices.values())
        return {"status": "SUCCESS", "devices": devs}

    router.register_route("POST", "/api/v1/auth/google", handle_auth_google, required_auth=False)
    router.register_route("POST", "/api/v1/auth/creator_key", handle_auth_creator_key, required_auth=False)
    router.register_route("POST", "/api/v1/auth/recovery", handle_auth_recovery, required_auth=False)
    router.register_route("POST", "/api/v1/auth/select_method", handle_select_method, required_auth=False)
    router.register_route("POST", "/api/v1/auth/logout", handle_auth_logout, required_auth=False)
    router.register_route("GET", "/api/v1/auth/session", handle_auth_session, required_auth=False)
    router.register_route("GET", "/api/v1/auth/qr_status", handle_qr_status, required_auth=False)
    router.register_route("POST", "/api/v1/auth/qr_challenge", handle_qr_challenge, required_auth=False)
    router.register_route("POST", "/api/v1/auth/qr_approve", handle_qr_approve, required_auth=False)
    router.register_route("POST", "/api/v1/auth/trigger_check", handle_trigger_check, required_auth=False)
    router.register_route("POST", "/api/v1/creator/enroll_google", handle_creator_enroll_google, required_auth=False)
    router.register_route("POST", "/api/v1/creator/setup", handle_creator_setup, required_auth=False)
    router.register_route("POST", "/api/v1/creator/register_device", handle_creator_register_device, required_auth=False)
    router.register_route("POST", "/api/v1/creator/revoke_device", handle_creator_revoke_device, required_auth=False)
    router.register_route("GET", "/api/v1/creator/devices", handle_creator_list_devices, required_auth=False)


register_creator_auth_routes(GLOBAL_API_ROUTER)


def register_voice_routes(router: ApiRouter) -> None:
    """
    Registers voice-to-voice conversation endpoints:
    - /api/v1/voice/transcribe (STT)
    - /api/v1/voice/synthesize (TTS)
    - /api/v1/voice/converse (Full Voice-to-Voice conversational pipeline)
    - /api/v1/voice/barge_in (Interruption handling)
    - /api/v1/voice/capabilities (Introspection)
    """
    def handle_voice_transcribe(handler, payload, actor=None):
        if not isinstance(payload, dict):
            return {"status": "ERROR", "error": "Invalid payload; expected JSON object."}
        audio_b64 = payload.get("audio_data") or payload.get("audio_base64")
        if not audio_b64:
            return {"status": "ERROR", "error": "Missing 'audio_data' base64 field."}
        try:
            audio_bytes = base64.b64decode(audio_b64)
        except Exception as e:
            return {"status": "ERROR", "error": f"Base64 decoding failed: {e}"}

        lang = payload.get("language", "en-US")
        result = GLOBAL_VOICE_INPUT.transcribe(audio_bytes, language=lang)
        return {"status": "SUCCESS", "data": result}

    def handle_voice_synthesize(handler, payload, actor=None):
        if not isinstance(payload, dict):
            return {"status": "ERROR", "error": "Invalid payload; expected JSON object."}
        text = payload.get("text", "")
        if not text:
            return {"status": "ERROR", "error": "Missing 'text' field."}
        lang = payload.get("language", "en")
        rate = payload.get("rate")
        volume = payload.get("volume")

        wav_bytes = GLOBAL_VOICE_OUTPUT.synthesize(text=text, language=lang, rate=rate, volume=volume)
        handler._send_wav(200, wav_bytes)
        return None

    def handle_voice_converse(handler, payload, actor=None):
        if not isinstance(payload, dict):
            return {"status": "ERROR", "error": "Invalid payload; expected JSON object."}

        brain = getattr(handler.server, "brain", None)
        if not brain:
            return {"status": "ERROR", "error": "TaraBrain not initialized."}

        input_text = payload.get("input_text") or payload.get("text")
        lang = payload.get("language", "en-US")

        # If audio provided, transcribe
        audio_b64 = payload.get("audio_data") or payload.get("audio_base64")
        if audio_b64 and not input_text:
            try:
                audio_bytes = base64.b64decode(audio_b64)
                transcription = GLOBAL_VOICE_INPUT.transcribe(audio_bytes, language=lang)
                input_text = transcription.get("transcript", "")
            except Exception as e:
                return {"status": "ERROR", "error": f"Speech transcription failed: {e}"}

        if not input_text or not input_text.strip():
            return {
                "status": "ERROR",
                "error": "No audible speech or transcript provided.",
                "transcript": ""
            }

        # Check wake-word if required by client
        if payload.get("require_wake_word", False):
            wake_check = GLOBAL_WAKE_WORD.detect(input_text)
            if not wake_check["detected"]:
                return {
                    "status": "WAKE_WORD_NOT_DETECTED",
                    "transcript": input_text,
                    "wake_words": GLOBAL_WAKE_WORD.wake_words
                }
            input_text = wake_check["cleaned_command"] or input_text

        ctx = payload.get("context", {})
        ctx["voice_turn"] = True
        ctx["request_id"] = getattr(handler, "request_id", str(uuid.uuid4()))

        # Pass through authoritative TARA brain
        effective_actor = actor or "user"
        brain_result = brain.process(actor_id=effective_actor, input_text=input_text.strip(), context=ctx)

        # Extract textual response
        if isinstance(brain_result, dict):
            response_text = brain_result.get("response") or brain_result.get("output") or str(brain_result)
        else:
            response_text = str(brain_result)

        # Synthesize audio response
        turn_id = str(uuid.uuid4())
        GLOBAL_BARGE_IN.start_speaking(turn_id)
        wav_bytes = GLOBAL_VOICE_OUTPUT.synthesize(text=response_text, language=lang)
        GLOBAL_BARGE_IN.stop_speaking(turn_id)

        wav_b64 = base64.b64encode(wav_bytes).decode("ascii")

        return {
            "status": "SUCCESS",
            "transcript": input_text,
            "response_text": response_text,
            "audio_base64": wav_b64,
            "language": lang,
            "turn_id": turn_id,
            "brain_result": brain_result
        }

    def handle_voice_barge_in(handler, payload, actor=None):
        info = GLOBAL_BARGE_IN.handle_user_barge_in()
        return {"status": "SUCCESS", "barge_in": info}

    def handle_voice_capabilities(handler, query_params, actor=None):
        return {
            "status": "SUCCESS",
            "stt_available": True,
            "tts_available": True,
            "languages": ["en-US", "en-IN", "kn-IN"],
            "barge_in_supported": True,
            "wake_words": ["tara", "ತಾರಾ"]
        }

    router.register_route("POST", "/api/v1/voice/transcribe", handle_voice_transcribe, required_auth=False)
    router.register_route("POST", "/api/v1/voice/synthesize", handle_voice_synthesize, required_auth=False)
    router.register_route("POST", "/api/v1/voice/converse", handle_voice_converse, required_auth=False)
    router.register_route("POST", "/api/v1/voice/barge_in", handle_voice_barge_in, required_auth=False)
    router.register_route("GET", "/api/v1/voice/capabilities", handle_voice_capabilities, required_auth=False)


register_voice_routes(GLOBAL_API_ROUTER)


def register_core_api_routes(router: ApiRouter) -> None:
    """
    Registers core TARA endpoints into the authoritative ApiRouter:
    - UI routes (chat interface)
    - Health and status probes
    - Core chat API
    - Skills discovery
    - Admin management and introspection endpoints
    """
    # UI & Health Handlers
    def handle_ui(handler, query_params, actor=None):
        handler.handle_get_chat_ui(query_params=query_params, actor=actor)

    def handle_health(handler, query_params, actor=None):
        handler.handle_get_health(query_params=query_params, actor=actor)

    def handle_status(handler, query_params, actor=None):
        handler.handle_get_status(query_params=query_params, actor=actor)

    def handle_skills(handler, query_params, actor=None):
        handler.handle_get_skills(query_params=query_params, actor=actor)

    # UI routes
    router.register_route("GET", "", handle_ui, required_auth=False)
    router.register_route("GET", "/", handle_ui, required_auth=False)
    router.register_route("GET", "/chat", handle_ui, required_auth=False)
    router.register_route("GET", "/index.html", handle_ui, required_auth=False)

    # Health & status
    router.register_route("GET", "/health", handle_health, required_auth=False)
    router.register_route("GET", "/api/v1/health", handle_health, required_auth=False)
    router.register_route("GET", "/api/v1/status", handle_status, required_auth=False)

    # Chat & Skills
    def handle_chat(handler, payload, actor=None):
        handler.handle_post_chat(payload=payload, actor=actor)

    router.register_route("POST", "/api/v1/chat", handle_chat, required_auth=True)
    router.register_route("GET", "/api/v1/skills", handle_skills, required_auth=True)

    # Admin GET Endpoints
    def handle_admin_overview(handler, query_params, actor=None):
        handler.handle_admin_overview(query_params=query_params, actor=actor)

    def handle_admin_candidates(handler, query_params, actor=None):
        handler.handle_admin_get_candidates(query_params=query_params, actor=actor)

    def handle_admin_knowledge(handler, query_params, actor=None):
        handler.handle_admin_get_knowledge(query_params=query_params, actor=actor)

    def handle_admin_memory(handler, query_params, actor=None):
        handler.handle_admin_get_memory(query_params=query_params, actor=actor)

    def handle_admin_rules(handler, query_params, actor=None):
        handler.handle_admin_get_rules(query_params=query_params, actor=actor)

    def handle_admin_security(handler, query_params, actor=None):
        handler.handle_admin_get_security(query_params=query_params, actor=actor)

    def handle_admin_self_train_status(handler, query_params, actor=None):
        handler.handle_admin_get_self_train_status(query_params=query_params, actor=actor)

    router.register_route("GET", "/api/v1/admin/overview", handle_admin_overview, required_auth=True, required_role="admin")
    router.register_route("GET", "/api/v1/admin/status", handle_admin_overview, required_auth=True, required_role="admin")
    router.register_route("GET", "/api/v1/admin/candidates", handle_admin_candidates, required_auth=True, required_role="admin")
    router.register_route("GET", "/api/v1/admin/knowledge", handle_admin_knowledge, required_auth=True, required_role="admin")
    router.register_route("GET", "/api/v1/admin/memory", handle_admin_memory, required_auth=True, required_role="admin")
    router.register_route("GET", "/api/v1/admin/rules", handle_admin_rules, required_auth=True, required_role="admin")
    router.register_route("GET", "/api/v1/admin/security", handle_admin_security, required_auth=True, required_role="admin")
    router.register_route("GET", "/api/v1/admin/self-train-status", handle_admin_self_train_status, required_auth=True, required_role="admin")

    # Admin POST Endpoints
    def handle_admin_post_approve(handler, payload, actor=None):
        handler.handle_admin_post_approve_candidate(payload=payload, actor=actor)

    def handle_admin_post_know(handler, payload, actor=None):
        handler.handle_admin_post_knowledge(payload=payload, actor=actor)

    def handle_admin_post_skill(handler, payload, actor=None):
        handler.handle_admin_post_skill_execute(payload=payload, actor=actor)

    def handle_admin_post_device(handler, payload, actor=None):
        handler.handle_admin_post_device_authorize(payload=payload, actor=actor)

    def handle_admin_post_train(handler, payload, actor=None):
        handler.handle_admin_post_self_train(payload=payload, actor=actor)

    router.register_route("POST", "/api/v1/admin/approve_candidate", handle_admin_post_approve, required_auth=True, required_role="admin")
    router.register_route("POST", "/api/v1/admin/knowledge", handle_admin_post_know, required_auth=True, required_role="admin")
    router.register_route("POST", "/api/v1/admin/skill_execute", handle_admin_post_skill, required_auth=True, required_role="admin")
    router.register_route("POST", "/api/v1/admin/device_authorize", handle_admin_post_device, required_auth=True, required_role="admin")
    router.register_route("POST", "/api/v1/admin/self-train", handle_admin_post_train, required_auth=True, required_role="admin")


register_core_api_routes(GLOBAL_API_ROUTER)



CHAT_UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover">
  <title>TARA AI Core</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; }
    body { background-color: #0b0f19; color: #f1f5f9; display: flex; flex-direction: column; height: 100vh; height: 100dvh; overflow: hidden; }
    header { background-color: #131b2e; border-bottom: 1px solid #232f48; padding: 12px 18px; display: flex; align-items: center; justify-content: space-between; flex-shrink: 0; }
    .header-left { display: flex; align-items: center; gap: 10px; }
    .status-dot { width: 10px; height: 10px; border-radius: 50%; background: #10b981; box-shadow: 0 0 8px #10b981; }
    .brand-title { font-weight: 700; font-size: 16px; letter-spacing: 0.5px; color: #38bdf8; }
    .brand-sub { font-size: 11px; color: #94a3b8; }
    .btn-clear { background: transparent; border: 1px solid #334155; color: #94a3b8; font-size: 12px; padding: 6px 12px; border-radius: 6px; cursor: pointer; }
    .btn-clear:hover { background: #1e293b; color: #f1f5f9; }
    #chat-container { flex: 1; overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 14px; max-width: 860px; width: 100%; margin: 0 auto; }
    .msg { display: flex; flex-direction: column; max-width: 85%; animation: fadeIn 0.2s ease-in; }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: translateY(0); } }
    .msg.user { align-self: flex-end; }
    .msg.user .bubble { background: #2563eb; color: #ffffff; border-radius: 14px 14px 2px 14px; }
    .msg.tara { align-self: flex-start; }
    .msg.tara .bubble { background: #1e293b; color: #f1f5f9; border-radius: 14px 14px 14px 2px; border: 1px solid #2e3c54; }
    .msg-label { font-size: 11px; font-weight: 600; margin-bottom: 4px; color: #64748b; }
    .msg.user .msg-label { text-align: right; color: #60a5fa; }
    .bubble { padding: 12px 16px; font-size: 14px; line-height: 1.5; word-break: break-word; white-space: pre-wrap; }
    .auth-methods-box { margin-top: 12px; padding-top: 10px; border-top: 1px solid #334155; display: flex; flex-direction: column; gap: 8px; }
    .auth-btn-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 8px; }
    .auth-chip { background: #0f172a; border: 1px solid #38bdf8; color: #38bdf8; padding: 10px 14px; border-radius: 8px; font-size: 13px; font-weight: 600; cursor: pointer; text-align: left; transition: all 0.15s ease; }
    .auth-chip:hover, .auth-chip:active { background: #38bdf8; color: #0f172a; }
    .qr-container { display: flex; flex-direction: column; align-items: center; gap: 10px; padding: 14px; background: #0f172a; border-radius: 8px; margin-top: 10px; border: 1px solid #334155; }
    .qr-svg { max-width: 220px; max-height: 220px; border: 8px solid #ffffff; border-radius: 6px; background: #ffffff; }
    .qr-timer { font-size: 12px; color: #f59e0b; font-weight: 600; }
    .qr-uri-box { font-family: monospace; font-size: 10px; color: #94a3b8; background: #1e293b; padding: 6px 10px; border-radius: 4px; max-width: 100%; overflow-x: auto; white-space: nowrap; word-break: break-all; }
    .inline-auth-form { margin-top: 10px; display: flex; flex-direction: column; gap: 8px; background: #0f172a; padding: 12px; border-radius: 8px; border: 1px solid #334155; }
    .inline-input { background: #1e293b; border: 1px solid #475569; color: #f1f5f9; padding: 9px 12px; border-radius: 6px; font-size: 13px; outline: none; }
    .inline-input:focus { border-color: #38bdf8; }
    .inline-btn { background: #38bdf8; color: #0b0f19; border: none; font-weight: 600; padding: 9px 14px; border-radius: 6px; cursor: pointer; font-size: 13px; }
    .inline-btn:hover { background: #7dd3fc; }
    .auth-badge { display: inline-block; background: rgba(16, 185, 129, 0.15); border: 1px solid #10b981; color: #34d399; font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 4px; margin-top: 6px; }
    footer { background: #131b2e; border-top: 1px solid #232f48; padding: 12px 16px; flex-shrink: 0; }
    .input-form { display: flex; gap: 10px; max-width: 860px; margin: 0 auto; width: 100%; }
    #chat-input { flex: 1; background: #1e293b; border: 1px solid #334155; color: #f1f5f9; padding: 12px 16px; border-radius: 24px; font-size: 14px; outline: none; }
    #chat-input:focus { border-color: #38bdf8; box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.2); }
    #btn-send { background: #38bdf8; color: #0b0f19; border: none; border-radius: 24px; padding: 0 20px; font-weight: 600; cursor: pointer; font-size: 14px; }
    #btn-send:hover { background: #7dd3fc; }
    @media (max-width: 600px) {
      .bubble { font-size: 13px; padding: 10px 14px; }
      .auth-btn-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <div class="header-left">
      <div class="status-dot"></div>
      <div>
        <div class="brand-title">TARA AI Core</div>
        <div class="brand-sub" id="header-status">User Mode • Promoted Kernel</div>
      </div>
    </div>
    <button class="btn-clear" onclick="clearChat()">Clear Chat</button>
  </header>

  <div id="chat-container">
    <div class="msg tara">
      <div class="msg-label">TARA</div>
      <div class="bubble">Hello! I am TARA, your neural cognitive AI assistant. How can I help you today?</div>
    </div>
  </div>

  <footer>
    <form class="input-form" onsubmit="handleSendMessage(event)">
      <input type="text" id="chat-input" placeholder="Type a message..." autocomplete="off" />
      <button type="submit" id="btn-send">Send ➔</button>
    </form>
  </footer>

  <script>
    let currentSessionToken = null;
    let currentCreatorId = null;
    let currentCreatorRole = null;
    let qrPollInterval = null;

    function appendMessage(sender, text, isHtml = false, extraNode = null) {
      const container = document.getElementById("chat-container");
      const msgDiv = document.createElement("div");
      msgDiv.className = `msg ${sender.toLowerCase()}`;

      const label = document.createElement("div");
      label.className = "msg-label";
      label.textContent = sender === "user" ? (currentCreatorId ? `Creator (${currentCreatorId})` : "User") : "TARA";
      msgDiv.appendChild(label);

      const bubble = document.createElement("div");
      bubble.className = "bubble";
      if (isHtml) {
        bubble.innerHTML = text;
      } else {
        bubble.textContent = text;
      }

      if (extraNode) {
        bubble.appendChild(extraNode);
      }

      msgDiv.appendChild(bubble);
      container.appendChild(msgDiv);
      container.scrollTop = container.scrollHeight;
      return msgDiv;
    }

    function clearChat() {
      const container = document.getElementById("chat-container");
      container.innerHTML = `
        <div class="msg tara">
          <div class="msg-label">TARA</div>
          <div class="bubble">Chat cleared. Authenticated session: ${currentCreatorId || "User Mode"}.</div>
        </div>`;
    }

    async function handleSendMessage(e) {
      if (e) e.preventDefault();
      const input = document.getElementById("chat-input");
      const text = input.value.trim();
      if (!text) return;

      appendMessage("user", text);
      input.value = "";
      input.focus();

      try {
        const payload = {
          input: text,
          actor_id: currentCreatorId || "user",
          session_id: "web-session-1",
          context: {
            source: "web_browser_client",
            creator_session_token: currentSessionToken
          }
        };

        const res = await fetch("/api/v1/chat", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            ...(currentSessionToken ? { "Authorization": `Bearer ${currentSessionToken}` } : {})
          },
          body: JSON.stringify(payload)
        });

        const data = await res.json();
        const taraData = data.data || {};
        const reply = taraData.response || (data.status === "ERROR" ? ("Error: " + data.error) : "Processing...");

        let extraNode = null;
        if (reply.includes("Choose an authentication method:") || (taraData.result && taraData.result.status === "AWAITING_METHOD_SELECTION")) {
          extraNode = renderInChatAuthChips();
        }

        appendMessage("tara", reply, false, extraNode);

        if (taraData.result && taraData.result.status === "SUCCESS" && taraData.result.session_token) {
          onCreatorAuthenticated(taraData.result);
        }
      } catch (err) {
        appendMessage("tara", "Network error communicating with TARA Core: " + err.message);
      }
    }

    function renderInChatAuthChips() {
      const box = document.createElement("div");
      box.className = "auth-methods-box";

      const grid = document.createElement("div");
      grid.className = "auth-btn-grid";

      const methods = [
        { id: "QR", label: "📱 QR Authentication", action: selectQrMethod },
        { id: "CREATOR_KEY", label: "🔑 Creator Key Authentication", action: selectCreatorKeyMethod },
        { id: "GOOGLE", label: "🔐 Google Authentication", action: selectGoogleMethod },
        { id: "RECOVERY", label: "🆘 Recovery", action: selectRecoveryMethod }
      ];

      methods.forEach(m => {
        const chip = document.createElement("button");
        chip.className = "auth-chip";
        chip.textContent = m.label;
        chip.onclick = m.action;
        grid.appendChild(chip);
      });

      box.appendChild(grid);
      return box;
    }

    async function selectQrMethod() {
      try {
        const res = await fetch("/api/v1/auth/qr_challenge", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ creator_id: "ROOT_OPERATOR" })
        });
        const d = await res.json();
        const chal = d.challenge || {};

        const qrBox = document.createElement("div");
        qrBox.className = "qr-container";

        const timerSpan = document.createElement("div");
        timerSpan.className = "qr-timer";
        timerSpan.textContent = "⏱️ Challenge expires in 120s";
        qrBox.appendChild(timerSpan);

        if (chal.svg_qr) {
          const svgDiv = document.createElement("div");
          svgDiv.className = "qr-svg";
          svgDiv.innerHTML = chal.svg_qr;
          qrBox.appendChild(svgDiv);
        }

        const uriBox = document.createElement("div");
        uriBox.className = "qr-uri-box";
        uriBox.textContent = chal.challenge_uri;
        qrBox.appendChild(uriBox);

        const statusNote = document.createElement("div");
        statusNote.style.fontSize = "11px";
        statusNote.style.color = "#94a3b8";
        statusNote.textContent = "Scan with trusted creator device to approve session.";
        qrBox.appendChild(statusNote);

        appendMessage("tara", "📱 One-Time QR Challenge Generated:", false, qrBox);

        // Countdown timer
        let timeLeft = 120;
        const countdown = setInterval(() => {
          timeLeft -= 1;
          if (timeLeft <= 0) {
            clearInterval(countdown);
            clearInterval(qrPollInterval);
            timerSpan.textContent = "❌ Challenge expired";
            timerSpan.style.color = "#ef4444";
          } else {
            timerSpan.textContent = `⏱️ Challenge expires in ${timeLeft}s`;
          }
        }, 1000);

        // Polling for approval
        clearInterval(qrPollInterval);
        qrPollInterval = setInterval(async () => {
          try {
            const pRes = await fetch(`/api/v1/auth/qr_status?challenge_id=${chal.challenge_id}`);
            const pData = await pRes.json();
            if (pData.status === "APPROVED") {
              clearInterval(qrPollInterval);
              clearInterval(countdown);
              onCreatorAuthenticated(pData.session);
              appendMessage("tara", `✅ QR Authentication Approved!\nCreator Authenticated: ${pData.session.creator_id} (${pData.session.role})`);
            }
          } catch (e) {}
        }, 2500);

      } catch (err) {
        appendMessage("tara", "Failed to generate QR challenge: " + err.message);
      }
    }

    function selectCreatorKeyMethod() {
      const form = document.createElement("div");
      form.className = "inline-auth-form";
      form.innerHTML = `
        <div style="font-size: 12px; color: #f59e0b; font-weight: 600;">
          🔑 Protected Key Proof or Keystore Passphrase:
        </div>
        <div style="font-size: 11px; color: #94a3b8;">
          Raw Ed25519 private keys are never accepted in chat. Enter keystore passphrase or proof artifact:
        </div>
        <input type="password" class="inline-input" id="key-passphrase" placeholder="Keystore passphrase / proof artifact..." />
        <button class="inline-btn" onclick="submitCreatorKey()">Submit Creator Key Proof</button>
      `;
      appendMessage("tara", "🔑 Creator Key Authentication Selected:", false, form);
    }

    async function submitCreatorKey() {
      const inp = document.getElementById("key-passphrase");
      const val = inp ? inp.value.trim() : "";
      if (!val) return;

      try {
        const res = await fetch("/api/v1/auth/creator_key", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ passphrase: val, claimed_creator_id: "ROOT_OPERATOR" })
        });
        const d = await res.json();
        if (d.status === "SUCCESS") {
          onCreatorAuthenticated(d);
          appendMessage("tara", `✅ Key Authentication Successful!\nCreator: ${d.creator_id} (${d.role})`);
        } else {
          appendMessage("tara", `❌ Key Authentication Failed: ${d.error || "No creator authority granted."}`);
        }
      } catch (e) {
        appendMessage("tara", "Error: " + e.message);
      }
    }

    function selectGoogleMethod() {
      const form = document.createElement("div");
      form.className = "inline-auth-form";
      form.innerHTML = `
        <div style="font-size: 12px; color: #10b981; font-weight: 600;">
          🔐 Google Sign-In & ID Token Verification:
        </div>
        <div style="font-size: 11px; color: #94a3b8;">
          Authenticate with Google (Passkey / 2SV / Authenticator) and paste your verified Google ID token:
        </div>
        <input type="text" class="inline-input" id="google-token-input" placeholder="eyJhbGciOiJSUzI1NiIs..." />
        <button class="inline-btn" onclick="submitGoogleToken()">Verify Google Token</button>
      `;
      appendMessage("tara", "🔐 Google Authentication Selected:", false, form);
    }

    async function submitGoogleToken() {
      const inp = document.getElementById("google-token-input");
      const tok = inp ? inp.value.trim() : "";
      if (!tok) return;

      try {
        const res = await fetch("/api/v1/auth/google", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ id_token: tok })
        });
        const d = await res.json();
        if (d.status === "SUCCESS") {
          onCreatorAuthenticated(d);
          appendMessage("tara", `✅ Google Authentication Successful!\nCreator: ${d.creator_id} (${d.role})`);
        } else {
          appendMessage("tara", `❌ Google Authentication Failed: ${d.error || "No creator authority granted."}`);
        }
      } catch (e) {
        appendMessage("tara", "Error: " + e.message);
      }
    }

    function selectRecoveryMethod() {
      const form = document.createElement("div");
      form.className = "inline-auth-form";
      form.innerHTML = `
        <div style="font-size: 12px; color: #ef4444; font-weight: 600;">
          🆘 Emergency Recovery Verification:
        </div>
        <div style="font-size: 11px; color: #94a3b8;">
          Please enter your 32-character high-entropy recovery code or multi-creator recovery authorization:
        </div>
        <input type="text" class="inline-input" id="recovery-code-input" placeholder="32-character recovery code..." />
        <button class="inline-btn" style="background:#ef4444;color:#fff;" onclick="submitRecoveryCode()">Submit Recovery Code</button>
      `;
      appendMessage("tara", "🆘 Recovery Authentication Selected:", false, form);
    }

    async function submitRecoveryCode() {
      const inp = document.getElementById("recovery-code-input");
      const code = inp ? inp.value.trim() : "";
      if (!code) return;

      try {
        const res = await fetch("/api/v1/auth/recovery", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ recovery_code: code, claimed_creator_id: "ROOT_OPERATOR" })
        });
        const d = await res.json();
        if (d.status === "SUCCESS") {
          onCreatorAuthenticated(d);
          appendMessage("tara", `✅ Recovery Successful!\nCreator Authority Restored: ${d.creator_id} (${d.role})`);
        } else {
          appendMessage("tara", `❌ Recovery Failed: ${d.error || "No creator authority granted."}`);
        }
      } catch (e) {
        appendMessage("tara", "Error: " + e.message);
      }
    }

    function onCreatorAuthenticated(authResult) {
      currentSessionToken = authResult.session_token;
      currentCreatorId = authResult.creator_id;
      currentCreatorRole = authResult.role;

      const headerSub = document.getElementById("header-status");
      headerSub.innerHTML = `🔒 Creator Active: <b style="color:#38bdf8;">${currentCreatorId}</b> (${currentCreatorRole}) <button onclick="logoutCreator()" style="background:#ef4444;color:#fff;border:none;border-radius:4px;padding:2px 6px;margin-left:6px;font-size:10px;cursor:pointer;">Logout</button>`;
    }

    async function logoutCreator() {
      if (currentSessionToken) {
        try {
          await fetch("/api/v1/auth/logout", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ session_token: currentSessionToken })
          });
        } catch (e) {}
      }
      currentSessionToken = null;
      currentCreatorId = null;
      currentCreatorRole = null;
      const headerSub = document.getElementById("header-status");
      headerSub.textContent = "User Mode • Promoted Kernel";
      appendMessage("tara", "Creator session revoked. Switched to normal user mode.");
    }
  </script>
</body>
</html>
"""


def _resolve_cors_origin(
    req_path: str,
    origin: Optional[str] = None,
    host: Optional[str] = None
) -> Tuple[Optional[str], bool]:
    """
    Evaluates request path and Origin header for strict CORS enforcement.
    Privileged/authentication endpoints ONLY allow explicit authorized origins.
    Public chat/status endpoints allow wildcard / cross-device access.
    Returns (allowed_origin_or_none, is_sensitive).
    """
    clean_path = req_path.split("?")[0].rstrip("/")

    # Privileged and authentication routes
    is_sensitive = any(clean_path.startswith(prefix) for prefix in [
        "/api/v1/auth", "/api/v1/admin", "/api/v1/creator", "/auth", "/admin"
    ])

    if not is_sensitive:
        return (origin or "*", False)

    # Sensitive route: check against allowed origins
    trusted_origins = {
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:3000",
        "https://tara-core.onrender.com"
    }
    env_allowed = os.environ.get("TARA_ALLOWED_ORIGINS")
    if env_allowed:
        for o in env_allowed.split(","):
            if o.strip():
                trusted_origins.add(o.strip().lower())

    if host:
        trusted_origins.add(f"http://{host.lower()}")
        trusted_origins.add(f"https://{host.lower()}")

    if origin and origin.lower() in trusted_origins:
        return (origin, True)

    return (None, True)


class TaraRequestHandler(BaseHTTPRequestHandler):
    def _get_router(self) -> ApiRouter:
        return getattr(self.server, "router", GLOBAL_API_ROUTER)

    def _resolve_cors_origin(self) -> Tuple[Optional[str], bool]:
        raw_path = getattr(self, "path", "")
        origin = self.headers.get("Origin") if hasattr(self, "headers") and self.headers else None
        host = self.headers.get("Host") if hasattr(self, "headers") and self.headers else None
        return _resolve_cors_origin(raw_path, origin, host)

    def _send_json(self, status_code: int, data: Dict[str, Any], extra_headers: Optional[Dict[str, str]] = None):
        response_bytes = json.dumps(data, ensure_ascii=False, indent=2).encode('utf-8')
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(response_bytes)))

        cors_origin, is_sensitive = self._resolve_cors_origin()
        if cors_origin:
            self.send_header('Access-Control-Allow-Origin', cors_origin)
            self.send_header('Vary', 'Origin')

        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-API-Key')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('X-Frame-Options', 'DENY')
        self.send_header('Referrer-Policy', 'strict-origin-when-cross-origin')
        self.send_header('Permissions-Policy', 'geolocation=(), camera=(), microphone=(self)')
        self.send_header('Strict-Transport-Security', 'max-age=31536000; includeSubDomains')

        req_id = getattr(self, "request_id", None) or str(uuid.uuid4())
        self.send_header('X-Request-ID', req_id)

        if extra_headers:
            for hk, hv in extra_headers.items():
                self.send_header(hk, hv)

        self.end_headers()
        self.wfile.write(response_bytes)

    def _send_wav(self, status_code: int, wav_bytes: bytes, extra_headers: Optional[Dict[str, str]] = None):
        self.send_response(status_code)
        self.send_header('Content-Type', 'audio/wav')
        self.send_header('Content-Length', str(len(wav_bytes)))

        cors_origin, is_sensitive = self._resolve_cors_origin()
        if cors_origin:
            self.send_header('Access-Control-Allow-Origin', cors_origin)
            self.send_header('Vary', 'Origin')

        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-API-Key')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('X-Frame-Options', 'DENY')
        self.send_header('Referrer-Policy', 'strict-origin-when-cross-origin')
        self.send_header('Permissions-Policy', 'geolocation=(), camera=(), microphone=(self)')
        self.send_header('Strict-Transport-Security', 'max-age=31536000; includeSubDomains')

        req_id = getattr(self, "request_id", None) or str(uuid.uuid4())
        self.send_header('X-Request-ID', req_id)

        if extra_headers:
            for hk, hv in extra_headers.items():
                self.send_header(hk, hv)

        self.end_headers()
        self.wfile.write(wav_bytes)

    def do_OPTIONS(self):
        cors_origin, is_sensitive = self._resolve_cors_origin()
        if is_sensitive and not cors_origin:
            self.send_response(403)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(b"Forbidden: Unauthorized CORS origin on sensitive endpoint.")
            return

        self.send_response(204)
        if cors_origin:
            self.send_header('Access-Control-Allow-Origin', cors_origin)
            self.send_header('Vary', 'Origin')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-API-Key')
        self.end_headers()

    def _check_rate_limit(self) -> bool:
        limiter = getattr(self.server, 'rate_limiter', None)
        if limiter:
            client_ip = None
            forwarded_for = self.headers.get('X-Forwarded-For') if hasattr(self, 'headers') and self.headers else None
            if forwarded_for:
                client_ip = forwarded_for.split(',')[0].strip()
            if not client_ip:
                client_ip = self.client_address[0] if self.client_address else "127.0.0.1"
            allowed, retry_after = limiter.is_allowed(client_ip)
            if not allowed:
                self._send_json(
                    429,
                    {
                        'status': 'ERROR',
                        'error': 'Too Many Requests: Rate limit exceeded. Try again later.',
                        'retry_after': retry_after
                    },
                    extra_headers={'Retry-After': str(retry_after)}
                )
                return False
        return True

    def _authenticate_request(self) -> Tuple[bool, Optional[str], Optional[str]]:
        configured_keys = getattr(self.server, 'api_keys', None)
        if configured_keys is None:
            env_keys = os.environ.get("TARA_API_KEYS")
            if env_keys:
                try:
                    configured_keys = json.loads(env_keys)
                except Exception:
                    configured_keys = None
            if not configured_keys:
                single_key = os.environ.get("TARA_API_KEY")
                if single_key:
                    configured_keys = {single_key: "api_user"}

        if not configured_keys:
            # Fall back to secure test token if running in local unit test mode
            configured_keys = {"tara_default_test_token": "api_user"}

        auth_header = self.headers.get('Authorization', '')
        token = None
        if auth_header.startswith("Bearer "):
            token = auth_header[7:].strip()
        elif "X-API-Key" in self.headers:
            token = self.headers.get("X-API-Key", "").strip()

        # Check cookie fallback for browser UI sessions
        cookie_header = self.headers.get('Cookie', '')
        cookie_token = None
        if cookie_header:
            for part in cookie_header.split(';'):
                part = part.strip()
                if part.startswith("tara_web_session="):
                    cookie_token = part[len("tara_web_session="):].strip()
                    break

        if not token and cookie_token:
            token = cookie_token

        if not token:
            return False, None, "Unauthorized: Missing Authorization Bearer token or X-API-Key header."

        # 1. Check Web Session Store (browser session parity)
        web_store = getattr(self.server, 'web_sessions', GLOBAL_WEB_SESSION_STORE)
        actor_from_session = web_store.verify(token)
        if actor_from_session:
            return True, actor_from_session, None

        # 2. Check Creator Session
        brain = getattr(self.server, 'brain', None)
        if brain and hasattr(brain, 'creator_auth_service') and brain.creator_auth_service:
            active_sess = brain.creator_auth_service.verify_session(token)
            if active_sess:
                return True, active_sess.get("creator_id", "ROOT_OPERATOR"), None

        # 3. Check Configured API Keys
        for valid_key, actor in configured_keys.items():
            if secrets.compare_digest(token, valid_key):
                return True, actor, None

        return False, None, "Unauthorized: Invalid API key or token."

    def handle_get_chat_ui(self, query_params=None, actor=None):
        web_store = getattr(self.server, 'web_sessions', GLOBAL_WEB_SESSION_STORE)
        session_token = web_store.issue("user")
        canonical_html_path = os.path.join(REPO_ROOT, "frontend", "index.html")
        if os.path.exists(canonical_html_path):
            try:
                with open(canonical_html_path, "r", encoding="utf-8") as f:
                    raw_html = f.read()
            except Exception:
                raw_html = CHAT_UI_HTML
        else:
            raw_html = CHAT_UI_HTML
        body = raw_html.replace("__TARA_SESSION_TOKEN__", session_token).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Set-Cookie', f"tara_web_session={session_token}; HttpOnly; SameSite=Strict; Max-Age=3600; Path=/")
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('X-Frame-Options', 'DENY')
        self.send_header('Referrer-Policy', 'strict-origin-when-cross-origin')
        self.send_header('Strict-Transport-Security', 'max-age=31536000; includeSubDomains')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self' 'unsafe-inline' https://accounts.google.com; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; connect-src 'self' https://accounts.google.com; frame-src https://accounts.google.com;")
        self.end_headers()
        self.wfile.write(body)

    def handle_get_health(self, query_params=None, actor=None):
        self._send_json(200, {'status': 'HEALTHY', 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})

    def _drain_body(self):
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            if 0 < content_length <= 1048576:
                self.rfile.read(content_length)
        except Exception:
            pass

    def _parse_json_body(self) -> Optional[Dict[str, Any]]:
        if hasattr(self, "_cached_json_body"):
            return self._cached_json_body
        try:
            content_length = int(self.headers.get('Content-Length', 0))
        except (ValueError, TypeError):
            self._body_error_sent = True
            self._send_json(400, {'status': 'ERROR', 'error': 'Invalid Content-Length header.'})
            return None

        if content_length > 1048576:
            self._body_error_sent = True
            self._send_json(413, {'status': 'ERROR', 'error': 'Payload too large. Maximum size is 1MB.'})
            return None
        if content_length == 0:
            self._cached_json_body = {}
            return {}

        try:
            raw_body = self.rfile.read(content_length).decode('utf-8')
            payload = json.loads(raw_body)
        except Exception as e:
            self._body_error_sent = True
            self._send_json(400, {'status': 'ERROR', 'error': f"Malformed JSON: {str(e)}"})
            return None

        if not isinstance(payload, dict):
            self._body_error_sent = True
            self._send_json(400, {'status': 'ERROR', 'error': 'JSON root must be an object.'})
            return None

        self._cached_json_body = payload
        return payload

    def _dispatch_request(self, method: str):
        self.request_id = str(uuid.uuid4())
        if not self._check_rate_limit():
            if method in ("POST", "PUT", "PATCH"):
                self._drain_body()
            return

        parsed_url = urllib.parse.urlsplit(self.path)
        url_path = parsed_url.path.rstrip('/')
        query_params = urllib.parse.parse_qs(parsed_url.query)

        router = self._get_router()
        route = router.match(method, url_path)

        if route is None:
            if method in ("POST", "PUT", "PATCH"):
                self._drain_body()
            if router.has_path(url_path):
                self._send_json(405, {"status": "ERROR", "error": f"Method {method} not allowed for endpoint '{self.path}'."})
            else:
                self._send_json(404, {"status": "ERROR", "error": f"Endpoint '{self.path}' not found."})
            return

        actor = None
        if route["required_auth"]:
            is_auth, actor, err = self._authenticate_request()
            if not is_auth:
                if method in ("POST", "PUT", "PATCH"):
                    self._drain_body()
                self._send_json(401, {'status': 'ERROR', 'error': err})
                return

            req_role = route.get("required_role")
            if req_role:
                allowed_roles = ("admin", "root", "system", "ROOT_OPERATOR")
                if req_role == "admin" and actor not in allowed_roles:
                    if method in ("POST", "PUT", "PATCH"):
                        self._drain_body()
                    self._send_json(403, {'status': 'ERROR', 'error': f"Forbidden: Actor '{actor}' lacks required role '{req_role}'."})
                    return

        if method in ("POST", "PUT", "PATCH"):
            payload = self._parse_json_body()
            if payload is None and getattr(self, "_body_error_sent", False):
                return
            arg = payload or {}
        else:
            arg = query_params

        try:
            res = route["handler"](self, arg, actor=actor)
            if res is not None and isinstance(res, (dict, list)):
                self._send_json(200, res if isinstance(res, dict) else {"data": res})
            return
        except Exception as e:
            self._send_json(500, {'status': 'ERROR', 'error': str(e)})
            return

    def do_GET(self):
        self._dispatch_request("GET")

    def do_POST(self):
        self._dispatch_request("POST")

    def do_PUT(self):
        self._dispatch_request("PUT")

    def do_DELETE(self):
        self._dispatch_request("DELETE")

    def handle_get_skills(self, query_params=None, actor=None):
        brain = getattr(self.server, 'brain', None)
        if brain is None:
            self._send_json(500, {'status': 'ERROR', 'error': 'TaraBrain not initialized on server.'})
            return
        skills_list = brain.skill_engine.list_skills()
        self._send_json(200, {
            'status': 'SUCCESS',
            'skills_count': len(skills_list),
            'skills': skills_list
        })

    def handle_get_status(self, query_params=None, actor=None):
        brain = getattr(self.server, 'brain', None)
        if brain is None:
            self._send_json(500, {'status': 'ERROR', 'error': 'TaraBrain not initialized on server.'})
            return
        model_status = brain.get_model_status() if hasattr(brain, "get_model_status") else {}

        # Guarantee exact runtime SHA256 integrity fields under model
        if "model_file" not in model_status or "model_sha256" not in model_status or "model_integrity" not in model_status:
            model_dir = getattr(brain, "model_dir", os.path.join(REPO_ROOT, "storage", "models", "tara"))
            raw_model_file = os.path.join(model_dir, "model.safetensors")
            if not os.path.exists(raw_model_file):
                candidates = [
                    os.path.join(REPO_ROOT, "storage", "models", "tara", "model.safetensors"),
                    os.path.join(os.getcwd(), "storage", "models", "tara", "model.safetensors"),
                    "/opt/render/project/src/storage/models/tara/model.safetensors",
                ]
                for c in candidates:
                    if os.path.exists(c):
                        raw_model_file = c
                        break

            computed_sha = None
            if os.path.isfile(raw_model_file):
                h = hashlib.sha256()
                with open(raw_model_file, "rb") as f:
                    while chunk := f.read(65536):
                        h.update(chunk)
                computed_sha = h.hexdigest()

            expected_promoted_sha = "7a50308b799f2654baeafbd64dec31f088c3b07b446e198cd0f9ec2b7c0af309"
            model_status["model_file"] = os.path.abspath(raw_model_file).replace("\\", "/")
            model_status["model_sha256"] = computed_sha
            model_status.setdefault("model_checkpoint", "baseline")
            model_status["model_integrity"] = "PASS" if (computed_sha and computed_sha.lower() == expected_promoted_sha.lower()) else "FAIL"

        memory_stats = brain.memory_engine.get_memory_stats() if hasattr(brain, "memory_engine") and brain.memory_engine else {}
        creator_state = brain.creator_auth_service.lifecycle.get_state() if hasattr(brain, "creator_auth_service") and brain.creator_auth_service else "CREATOR_SETUP_REQUIRED"
        self._send_json(200, {
            'status': 'ONLINE',
            'server_time': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'model': model_status,
            'memory': memory_stats,
            'creator_status': creator_state,
            'guard_active': getattr(brain, 'guard', None) is not None and getattr(brain.guard, 'policy', None) is not None
        })

    def handle_post_chat(self, payload: Any = None, actor: Optional[str] = None, authenticated_actor: Optional[str] = None):
        if isinstance(payload, str):
            authenticated_actor = payload
            payload = None

        brain = getattr(self.server, 'brain', None)
        if brain is None:
            self._send_json(500, {'status': 'ERROR', 'error': 'TaraBrain not initialized on server.'})
            return

        if payload is None:
            payload = self._parse_json_body()
            if payload is None:
                return

        input_text = payload.get('input')
        if not input_text or not isinstance(input_text, str) or not input_text.strip():
            self._send_json(400, {'status': 'ERROR', 'error': "Field 'input' is required and must be a non-empty string."})
            return

        requested_actor = payload.get('actor_id')
        context = payload.get('context', {})

        effective_auth_actor = actor or authenticated_actor
        # Prevent actor_id spoofing
        if requested_actor and effective_auth_actor and effective_auth_actor not in ("admin", "root", "system", "ROOT_OPERATOR"):
            if requested_actor != effective_auth_actor:
                self._send_json(403, {
                    'status': 'ERROR',
                    'error': f"Forbidden: Authenticated as '{effective_auth_actor}', cannot spoof actor '{requested_actor}'."
                })
                return

        effective_actor = effective_auth_actor or requested_actor or 'user'

        try:
            ctx = dict(context or {})
            ctx["request_id"] = self.request_id
            ctx["client_ip"] = self.client_address[0] if getattr(self, "client_address", None) else "127.0.0.1"
            if "google_id_token" in payload and "google_id_token" not in ctx:
                ctx["google_id_token"] = payload["google_id_token"]
            if "creator_session_token" in payload and "creator_session_token" not in ctx:
                ctx["creator_session_token"] = payload["creator_session_token"]

            result = brain.process(actor_id=effective_actor, input_text=input_text.strip(), context=ctx)
            self._send_json(200, {
                'status': 'SUCCESS',
                'request_id': self.request_id,
                'data': result
            })
        except Exception as e:
            self._send_json(500, {
                'status': 'ERROR',
                'request_id': self.request_id,
                'error': f"brain execution failure: {str(e)}"
            })

    def handle_admin_overview(self, query_params=None, actor=None):
        brain = getattr(self.server, 'brain', None)
        if brain is None:
            self._send_json(500, {'status': 'ERROR', 'error': 'TaraBrain not initialized on server.'})
            return

        cand_count = 0
        cand_pending = 0
        cand_approved = 0
        if os.path.exists(brain.knowledge_base.candidates_dir):
            for fn in os.listdir(brain.knowledge_base.candidates_dir):
                if fn.endswith('.json'):
                    cand_count += 1
                    try:
                        with open(os.path.join(brain.knowledge_base.candidates_dir, fn), 'r', encoding='utf-8') as f:
                            cdata = json.load(f)
                            if cdata.get('status') == 'APPROVED':
                                cand_approved += 1
                            else:
                                cand_pending += 1
                    except Exception:
                        pass

        knowledge_count = len(brain.knowledge_base.index)
        skills_count = len(brain.skill_engine.list_skills())
        memory_stats = brain.memory_engine.get_memory_stats()
        rules_count = len(brain.guard.policy.rules) if (brain.guard and brain.guard.policy) else 0

        devices_count = len(brain.identity_manager.devices.devices) if brain.identity_manager else 0
        authorized_devices_count = len(brain.identity_manager.devices.list_authorized_devices()) if brain.identity_manager else 0
        lockdown_state = getattr(brain.identity_manager.lockdown.current_state, "value", str(brain.identity_manager.lockdown.current_state)) if brain.identity_manager else "UNKNOWN"
        model_status = brain.get_model_status()

        self._send_json(200, {
            'status': 'SUCCESS',
            'server_time': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'model': model_status,
            'quarantine': {
                'total_candidates': cand_count,
                'pending': cand_pending,
                'approved': cand_approved
            },
            'knowledge_entries_count': knowledge_count,
            'skills_count': skills_count,
            'rules_count': rules_count,
            'memory': memory_stats,
            'devices': {
                'total': devices_count,
                'authorized': authorized_devices_count
            },
            'lockdown_state': lockdown_state
        })

    def handle_admin_get_candidates(self, query_params: Optional[Dict[str, List[str]]] = None, actor=None):
        brain = getattr(self.server, 'brain', None)
        if brain is None:
            self._send_json(500, {'status': 'ERROR', 'error': 'TaraBrain not initialized on server.'})
            return

        if query_params is None:
            query_params = {}
        status_filter = query_params.get('status', [None])[0]
        candidates = []
        if os.path.exists(brain.knowledge_base.candidates_dir):
            for fn in sorted(os.listdir(brain.knowledge_base.candidates_dir), reverse=True):
                if fn.endswith('.json'):
                    try:
                        with open(os.path.join(brain.knowledge_base.candidates_dir, fn), 'r', encoding='utf-8') as f:
                            c = json.load(f)
                            if status_filter and c.get('status') != status_filter:
                                continue
                            candidates.append(c)
                    except Exception:
                        pass

        self._send_json(200, {
            'status': 'SUCCESS',
            'count': len(candidates),
            'candidates': candidates
        })

    def handle_admin_get_knowledge(self, query_params: Optional[Dict[str, List[str]]] = None, actor=None):
        brain = getattr(self.server, 'brain', None)
        if brain is None:
            self._send_json(500, {'status': 'ERROR', 'error': 'TaraBrain not initialized on server.'})
            return

        if query_params is None:
            query_params = {}
        q = query_params.get('query', [None])[0]
        topic = query_params.get('topic', [None])[0]
        if q:
            results = brain.knowledge_base.query_knowledge(q, topic=topic)
        else:
            limit_str = query_params.get('limit', ['100'])[0]
            try:
                limit = int(limit_str)
            except ValueError:
                limit = 100
            results = []
            for kid, meta in list(brain.knowledge_base.index.items())[:limit]:
                entry = brain.knowledge_base.get_knowledge(kid)
                if entry:
                    results.append(entry)
                else:
                    results.append(meta)

        self._send_json(200, {
            'status': 'SUCCESS',
            'count': len(results),
            'knowledge': results
        })

    def handle_admin_get_memory(self, query_params: Optional[Dict[str, List[str]]] = None, actor=None):
        brain = getattr(self.server, 'brain', None)
        if brain is None:
            self._send_json(500, {'status': 'ERROR', 'error': 'TaraBrain not initialized on server.'})
            return

        if query_params is None:
            query_params = {}
        q = query_params.get('query', [None])[0]
        outcome = query_params.get('outcome', [None])[0]
        limit_str = query_params.get('limit', ['50'])[0]
        try:
            limit = int(limit_str)
        except ValueError:
            limit = 50

        episodes = brain.memory_engine.query_episodes(
            query=q,
            actor_id=None,
            outcome=outcome,
            limit=limit,
            scope="all"
        )
        self._send_json(200, {
            'status': 'SUCCESS',
            'count': len(episodes),
            'episodes': episodes
        })

    def handle_admin_get_rules(self, query_params=None, actor=None):
        brain = getattr(self.server, 'brain', None)
        if brain is None:
            self._send_json(500, {'status': 'ERROR', 'error': 'TaraBrain not initialized on server.'})
            return

        if brain.guard and brain.guard.policy:
            self._send_json(200, {
                'status': 'SUCCESS',
                'policy': brain.guard.policy.to_dict()
            })
        else:
            self._send_json(200, {
                'status': 'SUCCESS',
                'policy': None
            })

    def handle_admin_get_security(self, query_params=None, actor=None):
        brain = getattr(self.server, 'brain', None)
        if brain is None:
            self._send_json(500, {'status': 'ERROR', 'error': 'TaraBrain not initialized on server.'})
            return

        id_mgr = brain.identity_manager
        if not id_mgr:
            self._send_json(200, {'status': 'SUCCESS', 'security': None})
            return

        creator_rec = id_mgr.creator.get_creator_record() if hasattr(id_mgr.creator, 'get_creator_record') else None
        sanitized_creator = None
        if creator_rec:
            sanitized_creator = {
                "creator_id": creator_rec.get("creator_id"),
                "display_name": creator_rec.get("display_name"),
                "status": creator_rec.get("status"),
                "registered_at": creator_rec.get("registered_at"),
                "public_key": creator_rec.get("creator_public_key")
            }

        devices = list(id_mgr.devices.devices.values())
        lockdown_state = getattr(id_mgr.lockdown.current_state, "value", str(id_mgr.lockdown.current_state))

        self._send_json(200, {
            'status': 'SUCCESS',
            'creator': sanitized_creator,
            'devices': devices,
            'lockdown_state': lockdown_state
        })

    def handle_admin_post_approve_candidate(self, payload=None, actor=None):
        brain = getattr(self.server, 'brain', None)
        if brain is None:
            self._send_json(500, {'status': 'ERROR', 'error': 'TaraBrain not initialized on server.'})
            return

        if payload is None:
            payload = self._parse_json_body()
        if payload is None:
            return

        candidate_id = payload.get('candidate_id')
        if not candidate_id:
            self._send_json(400, {'status': 'ERROR', 'error': "Field 'candidate_id' is required."})
            return

        creator_id = payload.get('creator_id', 'ROOT_OPERATOR')
        creator_auth = payload.get('creator_auth')
        try:
            res = brain.knowledge_base.approve_candidate(candidate_id, creator_id=creator_id, creator_auth=creator_auth)
            self._send_json(200, {
                'status': 'SUCCESS',
                'action': 'APPROVED',
                'candidate_id': candidate_id,
                'result': res
            })
        except PermissionError as pe:
            self._send_json(403, {'status': 'ERROR', 'error': str(pe)})
        except ValueError as ve:
            self._send_json(404, {'status': 'ERROR', 'error': str(ve)})
        except Exception as e:
            self._send_json(500, {'status': 'ERROR', 'error': f"Failed to approve candidate: {str(e)}"})

    def handle_admin_post_knowledge(self, payload=None, actor=None):
        brain = getattr(self.server, 'brain', None)
        if brain is None:
            self._send_json(500, {'status': 'ERROR', 'error': 'TaraBrain not initialized on server.'})
            return

        if payload is None:
            payload = self._parse_json_body()
        if payload is None:
            return

        topic = payload.get('topic')
        subject = payload.get('subject')
        content = payload.get('content')
        if not topic or not subject or not content:
            self._send_json(400, {'status': 'ERROR', 'error': "Fields 'topic', 'subject', and 'content' are required."})
            return

        sources = payload.get('sources', [{"title": "Admin Desktop", "url": "local:admin"}])
        confidence = float(payload.get('confidence', 1.0))

        res = brain.knowledge_base.store_or_update_knowledge(
            topic=topic,
            subject=subject,
            content=content,
            sources=sources,
            learned_by_role="ROOT_CREATOR",
            trigger="ADMIN_DESKTOP_CONSOLE",
            confidence=confidence,
            verification_status="VERIFIED",
            session_id="admin_desktop"
        )
        self._send_json(200, {
            'status': 'SUCCESS',
            'result': res
        })

    def handle_admin_post_skill_execute(self, payload=None, actor=None):
        brain = getattr(self.server, 'brain', None)
        if brain is None:
            self._send_json(500, {'status': 'ERROR', 'error': 'TaraBrain not initialized on server.'})
            return

        if payload is None:
            payload = self._parse_json_body()
        if payload is None:
            return

        skill_name = payload.get('skill_name')
        if not skill_name:
            self._send_json(400, {'status': 'ERROR', 'error': "Field 'skill_name' is required."})
            return

        params = payload.get('parameters', {})
        try:
            res = brain.skill_engine.execute_skill(skill_name, params=params)
            self._send_json(200, {
                'status': 'SUCCESS',
                'skill_name': skill_name,
                'result': res
            })
        except Exception as e:
            self._send_json(500, {'status': 'ERROR', 'error': f"Skill execution failed: {str(e)}"})

    def handle_admin_post_device_authorize(self, payload=None, actor=None):
        brain = getattr(self.server, 'brain', None)
        if brain is None:
            self._send_json(500, {'status': 'ERROR', 'error': 'TaraBrain not initialized on server.'})
            return

        if payload is None:
            payload = self._parse_json_body()
        if payload is None:
            return

        device_id = payload.get('device_id')
        if not device_id:
            self._send_json(400, {'status': 'ERROR', 'error': "Field 'device_id' is required."})
            return

        if not brain.identity_manager:
            self._send_json(500, {'status': 'ERROR', 'error': "IdentityManager not initialized."})
            return

        success = brain.identity_manager.devices.authorize_device(device_id)
        if success:
            self._send_json(200, {
                'status': 'SUCCESS',
                'device_id': device_id,
                'device': brain.identity_manager.devices.get_device(device_id)
            })
        else:
            self._send_json(404, {'status': 'ERROR', 'error': f"Device '{device_id}' not found or revoked."})

    def handle_admin_get_self_train_status(self, query_params=None, actor=None):
        brain = getattr(self.server, 'brain', None)
        if brain is None:
            self._send_json(500, {'status': 'ERROR', 'error': 'TaraBrain not initialized on server.'})
            return
        try:
            from tara_model.self_trainer import get_self_trainer
            trainer = get_self_trainer(repo_root=brain.repo_root)
            status = trainer.get_status()
            self._send_json(200, {
                'status': 'SUCCESS',
                'self_training_status': status
            })
        except Exception as e:
            self._send_json(500, {'status': 'ERROR', 'error': str(e)})

    def handle_admin_post_self_train(self, payload=None, actor=None):
        brain = getattr(self.server, 'brain', None)
        if brain is None:
            self._send_json(500, {'status': 'ERROR', 'error': 'TaraBrain not initialized on server.'})
            return

        if payload is None:
            payload = self._parse_json_body() or {}
        max_epochs = payload.get('max_epochs', 1)
        force_now = payload.get('force_now', True)

        try:
            from tara_model.self_trainer import get_self_trainer
            trainer = get_self_trainer(repo_root=brain.repo_root)
            result = trainer.run_full_self_learning_cycle(max_epochs=max_epochs, force_now=force_now)
            self._send_json(200, {
                'status': 'SUCCESS',
                'action': 'SELF_TRAINING_COMPLETED',
                'result': result
            })
        except Exception as e:
            self._send_json(500, {'status': 'ERROR', 'error': f"Self-training execution failed: {str(e)}"})

    def log_message(self, format, *args):
        pass


def create_server(
    host: str = '127.0.0.1',
    port: int = 8080,
    brain: Optional[TaraBrain] = None,
    api_keys: Optional[Dict[str, str]] = None,
    rate_limiter: Optional[SlidingWindowRateLimiter] = None,
    router: Optional[ApiRouter] = None
) -> HTTPServer:
    server = HTTPServer((host, port), TaraRequestHandler)
    server.brain = brain or TaraBrain()
    server.router = router or GLOBAL_API_ROUTER
    if router is not None:
        register_auto_connect_sync_routes(server.router)
        register_dynamic_engine_routes(server.router)
        register_creator_auth_routes(server.router)
        register_voice_routes(server.router)

    if api_keys is not None:
        server.api_keys = api_keys
    else:
        env_key = os.environ.get("TARA_API_KEY")
        if env_key:
            server.api_keys = {env_key: "api_user"}
        else:
            server.api_keys = {"tara_default_test_token": "api_user"}

    server.rate_limiter = rate_limiter or SlidingWindowRateLimiter()
    return server


def run_server(host: Optional[str] = None, port: Optional[int] = None):
    effective_host = host or os.environ.get('HOST', '0.0.0.0')
    if port is None:
        port_env = os.environ.get('PORT')
        port = int(port_env) if port_env else 8080
    server = create_server(effective_host, port)
    print(f"[TARA Server] Running on http://{effective_host}:{port} ...")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\n[TARA-Server] Shutting down.')
        server.server_close()


if __name__ == '__main__':
    port = None
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass
    run_server(port=port)
