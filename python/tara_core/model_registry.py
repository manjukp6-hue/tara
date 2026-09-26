"""
python/tara_core/model_registry.py

Open-Ended Model Version Registry for TARA.
Manages arbitrary model versions, checkpoints, cryptographic fingerprints (SHA-256),
tokenizer compatibility, capability alignment, rollback tracking, and active pointers.
No model count limit is assumed or enforced.
"""

import os
import json
import hashlib
import logging
import threading
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger("TARA.ModelRegistry")


@dataclass
class ModelVersionMetadata:
    version_id: str
    artifact_location: str
    weights_sha256: str
    config: Dict[str, Any] = field(default_factory=dict)
    parameter_count: int = 118080
    growth_type: Optional[str] = None
    parent_model_version: Optional[str] = None
    shard_count: int = 1
    tokenizer_vocab_size: int = 344
    compatible_capabilities: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    status: str = "registered"  # registered, active, rolled_back, deprecated
    registered_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version_id": self.version_id,
            "artifact_location": self.artifact_location,
            "weights_sha256": self.weights_sha256,
            "config": self.config,
            "parameter_count": self.parameter_count,
            "growth_type": self.growth_type,
            "parent_model_version": self.parent_model_version,
            "shard_count": self.shard_count,
            "tokenizer_vocab_size": self.tokenizer_vocab_size,
            "compatible_capabilities": self.compatible_capabilities,
            "metrics": self.metrics,
            "status": self.status,
            "registered_at": self.registered_at
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelVersionMetadata":
        return cls(
            version_id=d["version_id"],
            artifact_location=d["artifact_location"],
            weights_sha256=d.get("weights_sha256", ""),
            config=d.get("config", {}),
            parameter_count=d.get("parameter_count", 118080),
            growth_type=d.get("growth_type"),
            parent_model_version=d.get("parent_model_version"),
            shard_count=d.get("shard_count", 1),
            tokenizer_vocab_size=d.get("tokenizer_vocab_size", 344),
            compatible_capabilities=d.get("compatible_capabilities", []),
            metrics=d.get("metrics", {}),
            status=d.get("status", "registered"),
            registered_at=d.get("registered_at", datetime.now(timezone.utc).isoformat())
        )


class ModelRegistry:
    """
    Registry for open-ended model versions in TARA.
    Allows registering, inspecting, switching, and rolling back models without hardcoded version counts.
    """
    _instance: Optional["ModelRegistry"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self, registry_file: Optional[str] = None):
        if registry_file is None:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
            registry_file = os.path.join(repo_root, "storage", "models", "versions_manifest.json")
        self.registry_file = os.path.abspath(registry_file)
        os.makedirs(os.path.dirname(self.registry_file), exist_ok=True)
        self._versions: Dict[str, ModelVersionMetadata] = {}
        self._active_version_id: Optional[str] = None
        self._rollback_history: List[str] = []
        self._reg_lock = threading.RLock()
        self._load_registry()

    @classmethod
    def get_default(cls, registry_file: Optional[str] = None) -> "ModelRegistry":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(registry_file=registry_file)
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        with cls._lock:
            cls._instance = None

    def _load_registry(self) -> None:
        with self._reg_lock:
            if os.path.exists(self.registry_file):
                try:
                    with open(self.registry_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    self._active_version_id = data.get("active_version")
                    self._rollback_history = data.get("rollback_history", [])
                    for vid, vdata in data.get("versions", {}).items():
                        self._versions[vid] = ModelVersionMetadata.from_dict(vdata)
                except Exception as e:
                    logger.warning(f"Failed to read model registry file: {e}")

            # Register current promoted baseline model dynamically if not present
            baseline_safetensors = os.path.abspath(
                os.path.join(os.path.dirname(self.registry_file), "tara", "model.safetensors")
            )
            if not os.path.exists(baseline_safetensors):
                repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
                baseline_safetensors = os.path.join(repo_root, "storage", "models", "tara", "model.safetensors")

            if "TARA_BASELINE" not in self._versions and os.path.exists(baseline_safetensors):
                h = hashlib.sha256()
                with open(baseline_safetensors, "rb") as bf:
                    while chunk := bf.read(65536):
                        h.update(chunk)
                baseline_meta = ModelVersionMetadata(
                    version_id="TARA_BASELINE",
                    artifact_location="storage/models/tara",
                    weights_sha256=h.hexdigest(),
                    tokenizer_vocab_size=344,
                    status="active",
                    compatible_capabilities=["*"]
                )
                self._versions["TARA_BASELINE"] = baseline_meta
                if not self._active_version_id:
                    self._active_version_id = "TARA_BASELINE"
                self._save_registry()

    def _save_registry(self) -> None:
        with self._reg_lock:
            data = {
                "active_version": self._active_version_id,
                "rollback_history": self._rollback_history,
                "versions": {vid: v.to_dict() for vid, v in self._versions.items()}
            }
            with open(self.registry_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

    def register_version(self, metadata: ModelVersionMetadata) -> bool:
        with self._reg_lock:
            if not metadata.version_id:
                raise ValueError("Model version_id cannot be empty.")
            self._versions[metadata.version_id] = metadata
            self._save_registry()
            logger.info(f"Registered model version '{metadata.version_id}'")
            return True

    def register_model(
        self,
        model_id: str,
        model_path: str,
        architecture: str = "TaraForCausalLM",
        vocab_size: int = 344,
        parameter_count: int = 118080,
        growth_type: Optional[str] = None,
        parent_model_version: Optional[str] = None,
        shard_count: int = 1,
        metrics: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        weights_sha256: str = ""
    ) -> bool:
        """Convenience method to register model candidate or version."""
        cfg_path = os.path.join(model_path, "config.json")
        cfg_dict: Dict[str, Any] = {}
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg_dict = json.load(f)
                    if "total_parameters" in cfg_dict:
                        parameter_count = cfg_dict["total_parameters"]
                    if "vocab_size" in cfg_dict:
                        vocab_size = cfg_dict["vocab_size"]
            except Exception:
                pass

        if not weights_sha256:
            single_w = os.path.join(model_path, "model.safetensors")
            idx_w = os.path.join(model_path, "model.safetensors.index.json")
            if os.path.exists(single_w):
                h = hashlib.sha256()
                with open(single_w, "rb") as bf:
                    while chunk := bf.read(65536):
                        h.update(chunk)
                weights_sha256 = h.hexdigest()
            elif os.path.exists(idx_w):
                h = hashlib.sha256()
                with open(idx_w, "rb") as bf:
                    while chunk := bf.read(65536):
                        h.update(chunk)
                weights_sha256 = h.hexdigest()

        meta = ModelVersionMetadata(
            version_id=model_id,
            artifact_location=model_path,
            weights_sha256=weights_sha256,
            config=cfg_dict,
            parameter_count=parameter_count,
            growth_type=growth_type,
            parent_model_version=parent_model_version,
            shard_count=shard_count,
            tokenizer_vocab_size=vocab_size,
            compatible_capabilities=["*"],
            metrics=metrics or {},
            status="registered"
        )
        return self.register_version(meta)

    def set_active_model(self, model_id: str) -> bool:
        return self.set_active_version(model_id)

    def get_version(self, version_id: str) -> Optional[ModelVersionMetadata]:
        with self._reg_lock:
            return self._versions.get(version_id)

    def list_versions(self) -> List[ModelVersionMetadata]:
        with self._reg_lock:
            return list(self._versions.values())

    def get_active_version(self) -> Optional[ModelVersionMetadata]:
        with self._reg_lock:
            if self._active_version_id:
                return self._versions.get(self._active_version_id)
            return None

    def set_active_version(self, version_id: str) -> bool:
        with self._reg_lock:
            if version_id not in self._versions:
                raise ValueError(f"Version '{version_id}' is not registered in ModelRegistry.")

            if self._active_version_id and self._active_version_id != version_id:
                self._rollback_history.append(self._active_version_id)

            self._active_version_id = version_id
            self._versions[version_id].status = "active"
            self._save_registry()
            logger.info(f"Active model pointer updated to '{version_id}'")
            return True

    def rollback(self) -> Optional[str]:
        """Rolls back active model to previous version in history."""
        with self._reg_lock:
            if not self._rollback_history:
                return None
            prev = self._rollback_history.pop()
            if prev in self._versions:
                current = self._active_version_id
                if current and current in self._versions:
                    self._versions[current].status = "rolled_back"
                self._active_version_id = prev
                self._versions[prev].status = "active"
                self._save_registry()
                logger.info(f"Rolled back model from '{current}' to '{prev}'")
                return prev
            return None

    def check_tokenizer_compatibility(self, version_id: str, vocab_size: int = 344) -> bool:
        with self._reg_lock:
            ver = self._versions.get(version_id)
            if not ver:
                return False
            return ver.tokenizer_vocab_size == vocab_size
