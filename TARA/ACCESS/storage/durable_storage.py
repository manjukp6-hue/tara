"""
TARA/ACCESS/storage/durable_storage.py

Provider-Independent Durable Persistence Architecture for TARA.
Guarantees persistent mutable state across Render container restarts, deploys, and dyno sleep.

Classification:
Category A (Durable - Must Persist):
  - Creator Identity Record (operator_record.json)
  - Creator Registry (operators_registry.json)
  - Protected Authority Verifier (state_v1_verifier.json)
  - Creator Trigger Config (activation_config.json)
  - Recovery Configuration (restore_config.json)
  - Device Registry (devices.json)
  - Lockdown Security State (lockdown_state.json)
  - Registered Endpoints (registered_endpoints.json)
  - Compiled Root Policies (compiled_policy.json)
  - Episodic Memory (episodes.jsonl)
Category B (Safe to Regenerate):
  - Knowledge index & cache
  - Model checkpoints history
Category C (Temporary / Ephemeral):
  - In-flight request nonces & temporary cache

Fail-Closed Production Safety:
If deployed on Render (RENDER=true) or TARA_ENV=production without an external durable backend,
stateful creator mutations fail closed immediately with DurableStorageUnconfiguredError.
"""

import os
import sys
import json
import time
import hashlib
import threading
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple

CANONICAL_CREATOR_ID = "ROOT_OPERATOR"
CANONICAL_DISPLAY_NAME = "OPERATOR_ROOT"
CANONICAL_ROLE = "ROOT_CREATOR"


class DurableStorageError(Exception):
    """Base exception for durable storage failures."""
    pass


class DurableStorageUnconfiguredError(DurableStorageError):
    """Raised when running in production without a configured durable backend."""
    pass


class StorageCorruptionError(DurableStorageError):
    """Raised when corrupted data or integrity mismatch is detected."""
    pass


class ConcurrencyConflictError(DurableStorageError):
    """Raised on version mismatch during atomic compare-and-swap set."""
    pass


class DurableStorageProvider(ABC):
    """Abstract interface for provider-independent durable persistence."""

    @abstractmethod
    def get(self, key: str) -> Optional[bytes]:
        pass

    @abstractmethod
    def set(self, key: str, value: bytes, expected_version: Optional[int] = None) -> int:
        """Sets the key value. Returns the new version number."""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        pass

    @abstractmethod
    def list_keys(self, prefix: str = "") -> List[str]:
        pass

    @abstractmethod
    def health_check(self) -> Tuple[bool, str]:
        pass


class LocalFileStorageProvider(DurableStorageProvider):
    """
    Local file-based durable store with atomic rename, versioning,
    and SHA256 integrity checksum verification.
    """
    def __init__(self, base_dir: str):
        self.base_dir = os.path.abspath(base_dir)
        os.makedirs(self.base_dir, exist_ok=True)
        self._lock = threading.RLock()
        self._versions: Dict[str, int] = {}
        self._load_versions()

    def _get_key_path(self, key: str) -> str:
        safe_key = key.replace(":", "__").replace("/", "_").replace("\\", "_")
        return os.path.join(self.base_dir, f"{safe_key}.durable")

    def _load_versions(self):
        v_file = os.path.join(self.base_dir, "_versions_manifest.json")
        if os.path.exists(v_file):
            try:
                with open(v_file, "r", encoding="utf-8") as f:
                    self._versions = json.load(f)
            except Exception:
                self._versions = {}

    def _save_versions(self):
        v_file = os.path.join(self.base_dir, "_versions_manifest.json")
        tmp_file = f"{v_file}.tmp.{os.getpid()}"
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(self._versions, f, indent=2)
        os.replace(tmp_file, v_file)

    def get(self, key: str) -> Optional[bytes]:
        with self._lock:
            path = self._get_key_path(key)
            if not os.path.exists(path):
                return None
            try:
                with open(path, "rb") as f:
                    raw = f.read()
                # Envelope format: 64 bytes hex SHA256 + payload
                if len(raw) < 64:
                    raise StorageCorruptionError(f"Corrupted storage record for key: '{key}' (truncated)")
                expected_sha = raw[:64].decode("ascii")
                payload = raw[64:]
                actual_sha = hashlib.sha256(payload).hexdigest()
                if actual_sha != expected_sha:
                    raise StorageCorruptionError(
                        f"Checksum mismatch for key '{key}'! Expected {expected_sha}, got {actual_sha}"
                    )
                return payload
            except StorageCorruptionError:
                raise
            except Exception as e:
                raise DurableStorageError(f"Failed to read key '{key}': {e}")

    def set(self, key: str, value: bytes, expected_version: Optional[int] = None) -> int:
        with self._lock:
            current_v = self._versions.get(key, 0)
            if expected_version is not None and expected_version != current_v:
                raise ConcurrencyConflictError(
                    f"Version conflict on '{key}': expected {expected_version}, current {current_v}"
                )

            new_v = current_v + 1
            path = self._get_key_path(key)
            tmp_path = f"{path}.tmp.{os.getpid()}.{time.time_ns()}"

            sha = hashlib.sha256(value).hexdigest().encode("ascii")
            envelope = sha + value

            try:
                with open(tmp_path, "wb") as f:
                    f.write(envelope)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, path)
                self._versions[key] = new_v
                self._save_versions()
                return new_v
            except Exception as e:
                if os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass
                raise DurableStorageError(f"Failed to atomically persist key '{key}': {e}")

    def delete(self, key: str) -> bool:
        with self._lock:
            path = self._get_key_path(key)
            self._versions.pop(key, None)
            self._save_versions()
            if os.path.exists(path):
                os.unlink(path)
                return True
            return False

    def exists(self, key: str) -> bool:
        with self._lock:
            return os.path.exists(self._get_key_path(key))

    def list_keys(self, prefix: str = "") -> List[str]:
        with self._lock:
            keys = []
            for f in os.listdir(self.base_dir):
                if f.endswith(".durable"):
                    clean = f[:-8].replace("__", ":")
                    if clean.startswith(prefix):
                        keys.append(clean)
            return sorted(keys)

    def health_check(self) -> Tuple[bool, str]:
        try:
            test_key = "__health_probe__"
            self.set(test_key, b"ok")
            val = self.get(test_key)
            self.delete(test_key)
            if val == b"ok":
                return True, "Local durable storage is healthy and writable."
            return False, "Health check read returned corrupted data."
        except Exception as e:
            return False, f"Local durable storage unhealthy: {e}"


class SQLDurableStorageProvider(DurableStorageProvider):
    """
    SQL-backed durable storage provider compatible with PostgreSQL (Render/Supabase/Neon)
    and SQLite via standard Python DB-API 2.0.
    """
    def __init__(self, connection_url: str):
        self.connection_url = connection_url
        self._lock = threading.RLock()
        self._is_sqlite = connection_url.startswith("sqlite")
        self._init_db()

    def _get_connection(self):
        if self._is_sqlite:
            import sqlite3
            path = self.connection_url.replace("sqlite:///", "").replace("sqlite://", "")
            return sqlite3.connect(path, timeout=30.0)
        else:
            try:
                import psycopg2
                return psycopg2.connect(self.connection_url)
            except ImportError:
                import sqlite3
                # Fallback to local sqlite if psycopg2 is not installed in local environment
                fallback_path = os.path.join("storage", "tara_durable_fallback.db")
                os.makedirs(os.path.dirname(fallback_path), exist_ok=True)
                return sqlite3.connect(fallback_path, timeout=30.0)

    def _init_db(self):
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS tara_durable_state (
                        key TEXT PRIMARY KEY,
                        value BLOB,
                        version INTEGER NOT NULL DEFAULT 1,
                        checksum TEXT NOT NULL,
                        updated_at REAL NOT NULL
                    )
                """)
                conn.commit()
            finally:
                conn.close()

    def get(self, key: str) -> Optional[bytes]:
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT value, checksum FROM tara_durable_state WHERE key = ?", (key,))
                row = cursor.fetchone()
                if not row:
                    return None
                val, checksum = row
                val_bytes = bytes(val)
                actual_sha = hashlib.sha256(val_bytes).hexdigest()
                if actual_sha != checksum:
                    raise StorageCorruptionError(
                        f"Checksum mismatch for DB key '{key}': expected {checksum}, got {actual_sha}"
                    )
                return val_bytes
            finally:
                conn.close()

    def set(self, key: str, value: bytes, expected_version: Optional[int] = None) -> int:
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT version FROM tara_durable_state WHERE key = ?", (key,))
                row = cursor.fetchone()
                current_v = row[0] if row else 0

                if expected_version is not None and expected_version != current_v:
                    raise ConcurrencyConflictError(
                        f"Version mismatch for key '{key}': expected {expected_version}, current {current_v}"
                    )

                new_v = current_v + 1
                sha = hashlib.sha256(value).hexdigest()
                now = time.time()

                if current_v == 0:
                    cursor.execute(
                        "INSERT INTO tara_durable_state (key, value, version, checksum, updated_at) VALUES (?, ?, ?, ?, ?)",
                        (key, value, new_v, sha, now)
                    )
                else:
                    cursor.execute(
                        "UPDATE tara_durable_state SET value = ?, version = ?, checksum = ?, updated_at = ? WHERE key = ?",
                        (value, new_v, sha, now, key)
                    )
                conn.commit()
                return new_v
            finally:
                conn.close()

    def delete(self, key: str) -> bool:
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM tara_durable_state WHERE key = ?", (key,))
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()

    def exists(self, key: str) -> bool:
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT 1 FROM tara_durable_state WHERE key = ?", (key,))
                return cursor.fetchone() is not None
            finally:
                conn.close()

    def list_keys(self, prefix: str = "") -> List[str]:
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT key FROM tara_durable_state WHERE key LIKE ? ORDER BY key", (f"{prefix}%",))
                return [r[0] for r in cursor.fetchall()]
            finally:
                conn.close()

    def health_check(self) -> Tuple[bool, str]:
        try:
            test_key = "__sql_health__"
            self.set(test_key, b"ok")
            val = self.get(test_key)
            self.delete(test_key)
            return (val == b"ok", "SQL durable storage is healthy.")
        except Exception as e:
            return False, f"SQL durable storage check failed: {e}"


class DurableStorageManager:
    """
    Central Manager and Factory for TARA Durable State.
    Enforces Category A state classification, migration, and fail-closed security.
    """
    _instance: Optional["DurableStorageManager"] = None
    _lock = threading.RLock()

    CATEGORY_A_MAPPINGS = {
        "creator:record": "TARA/ACCESS/operator/operator_record.json",
        "creator:registry": "TARA/ACCESS/operator/operators_registry.json",
        "creator:trigger": "TARA/ACCESS/operator/activation_config.json",
        "creator:state_v1_verifier": "TARA/ACCESS/creator/state_v1_verifier.json",
        "creator:recovery_config": "TARA/ACCESS/restore/restore_config.json",
        "devices:registry": "TARA/ACCESS/devices/devices.json",
        "lockdown:state": "TARA/ACCESS/lockdown/lockdown_state.json",
        "endpoints:registered": "TARA/ACCESS/registered_endpoints.json",
        "rules:compiled_policy": "TARA/RULES/compiled_policy.json",
    }

    def __init__(self):
        self.env = os.environ.get("TARA_ENV", "development").lower()
        self.is_render = os.environ.get("RENDER", "false").lower() == "true"
        self.db_url = os.environ.get("TARA_DATABASE_URL") or os.environ.get("DATABASE_URL")
        self.provider: DurableStorageProvider = self._select_provider()

    def _select_provider(self) -> DurableStorageProvider:
        if self.db_url:
            return SQLDurableStorageProvider(self.db_url)

        # In production / Render without DB: fail-closed for mutations
        if self.is_render or self.env == "production":
            # Check if explicit persistent disk mount is provided
            persistent_dir = os.environ.get("TARA_PERSISTENT_DIR")
            if persistent_dir and os.path.exists(persistent_dir):
                return LocalFileStorageProvider(persistent_dir)
            # Unconfigured production backend
            return LocalFileStorageProvider("storage/durable_state")

        # Local dev / tests
        base = os.environ.get("TARA_DURABLE_DIR", "storage/durable_state")
        return LocalFileStorageProvider(base)

    @classmethod
    def get_instance(cls) -> "DurableStorageManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @classmethod
    def reset(cls):
        with cls._lock:
            cls._instance = None

    def is_production_unconfigured(self) -> bool:
        if (self.is_render or self.env == "production") and not self.db_url:
            persistent_dir = os.environ.get("TARA_PERSISTENT_DIR")
            if not (persistent_dir and os.path.exists(persistent_dir)):
                return True
        return False

    def get_record(self, state_key: str) -> Optional[Dict[str, Any]]:
        """Retrieves a parsed JSON dictionary for a Category A state record."""
        raw = self.provider.get(state_key)
        if raw is None:
            # Fallback to local filesystem for initial migration
            local_path = self.CATEGORY_A_MAPPINGS.get(state_key)
            if local_path and os.path.exists(local_path):
                with open(local_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return data
            return None

        try:
            return json.loads(raw.decode("utf-8"))
        except Exception as e:
            raise StorageCorruptionError(f"Failed to decode JSON for '{state_key}': {e}")

    def set_record(
        self,
        state_key: str,
        data: Dict[str, Any],
        expected_version: Optional[int] = None,
        mirror_local: bool = False
    ) -> int:
        """Persists a Category A state record with fail-closed checks."""
        if self.is_production_unconfigured():
            raise DurableStorageUnconfiguredError(
                "CRITICAL: TARA is running in production/Render without a durable external backend. "
                "Set TARA_DATABASE_URL or DATABASE_URL to enable durable state persistence."
            )

        # Invariant Protection: Never allow overwriting or resetting canonical creator
        if state_key in ("creator:record", "creator:registry"):
            creator_id = data.get("creator_id") or (data.get(CANONICAL_CREATOR_ID, {}).get("creator_id"))
            if state_key == "creator:record" and creator_id != CANONICAL_CREATOR_ID:
                raise StorageCorruptionError(
                    f"Invariant violation: Cannot overwrite canonical creator '{CANONICAL_CREATOR_ID}' with '{creator_id}'"
                )

        payload = json.dumps(data, indent=2).encode("utf-8")
        new_v = self.provider.set(state_key, payload, expected_version=expected_version)

        # Mirror write to local filesystem only if explicitly requested and not in test mode
        if mirror_local and self.env != "test":
            local_path = self.CATEGORY_A_MAPPINGS.get(state_key)
            if local_path:
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                tmp = f"{local_path}.tmp.{os.getpid()}"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                os.replace(tmp, local_path)

        return new_v

    def migrate_all_local_state(self) -> Dict[str, str]:
        """Atomically migrates local Category A files into the durable store."""
        results = {}
        for state_key, rel_path in self.CATEGORY_A_MAPPINGS.items():
            if os.path.exists(rel_path):
                try:
                    with open(rel_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    self.set_record(state_key, data)
                    results[state_key] = "MIGRATED_SUCCESS"
                except Exception as e:
                    results[state_key] = f"FAILED: {e}"
            else:
                results[state_key] = "FILE_NOT_FOUND"
        return results
