"""
TARA/ACCESS/devices/device_registry.py

Per-device cryptographic identity registry.
Every authorized device generates its own local key pair and receives a unique
Device ID (e.g. TARA-DEVICE-001). Creator private key is never shared across devices.
"""

import os
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

from ..crypto.ed25519 import Ed25519


class DeviceRegistry:
    """
    Tracks all authorized and pending devices in the TARA ecosystem.
    """
    def __init__(self, registry_path: Optional[str] = None):
        if registry_path is None:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
            registry_path = os.path.join(repo_root, "TARA", "ACCESS", "devices", "devices.json")
        self.registry_path = registry_path
        self.devices: Dict[str, Dict[str, Any]] = {}
        self.counter: int = 0

        if os.path.exists(self.registry_path):
            self.load()

    def _generate_device_id(self) -> str:
        self.counter += 1
        return f"TARA-DEVICE-{self.counter:03d}"

    def register_device(
        self,
        public_key_bytes: Optional[bytes] = None,
        device_name: str = "Primary Device",
        status: str = "PENDING",
        device_id: Optional[str] = None,
        public_key_hex: Optional[str] = None
    ) -> Dict[str, Any]:
        """Registers a new device with its own independent public key."""
        if device_id is None:
            device_id = self._generate_device_id()
        else:
            self.counter += 1

        if public_key_hex is None and public_key_bytes is not None:
            public_key_hex = public_key_bytes.hex()
        elif public_key_hex is None and public_key_bytes is None:
            raise ValueError("Either public_key_bytes or public_key_hex must be provided.")

        now = datetime.now(timezone.utc).isoformat()

        record = {
            "device_id": device_id,
            "device_name": device_name,
            "device_public_key": public_key_hex,
            "status": status,  # "AUTHORIZED", "PENDING", "REVOKED"
            "created_at": now,
            "last_verified": now,
            "key_version": 1,
            "authorization_state": {
                "biometric_confirmed": False,
                "creator_approved": (status == "AUTHORIZED"),
                "offline_allowed": True
            }
        }
        self.devices[device_id] = record
        self.save()
        return record

    def list_devices(self) -> List[Dict[str, Any]]:
        """Returns all registered devices."""
        return list(self.devices.values())

    def authorize_device(self, device_id: str, biometric_confirmed: bool = False) -> bool:
        """Marks a device as fully authorized."""
        if device_id not in self.devices:
            return False
        dev = self.devices[device_id]
        if dev["status"] == "REVOKED":
            return False
        dev["status"] = "AUTHORIZED"
        dev["last_verified"] = datetime.now(timezone.utc).isoformat()
        dev["authorization_state"]["creator_approved"] = True
        if biometric_confirmed:
            dev["authorization_state"]["biometric_confirmed"] = True
        self.save()
        return True

    def revoke_device(self, device_id: str, reason: str = "manual_revocation") -> bool:
        """Revokes authorization for a device."""
        if device_id not in self.devices:
            return False
        dev = self.devices[device_id]
        dev["status"] = "REVOKED"
        dev["revoked_at"] = datetime.now(timezone.utc).isoformat()
        dev["revocation_reason"] = reason
        dev["authorization_state"]["creator_approved"] = False
        self.save()
        return True

    def verify_device_challenge(self, device_id: str, challenge: bytes, signature: bytes) -> bool:
        """Cryptographically verifies a device signature on an authentication challenge."""
        dev = self.devices.get(device_id)
        if not dev or dev["status"] != "AUTHORIZED":
            return False

        pub_bytes = bytes.fromhex(dev["device_public_key"])
        is_valid = Ed25519.verify(pub_bytes, challenge, signature)
        if is_valid:
            dev["last_verified"] = datetime.now(timezone.utc).isoformat()
            self.save()
        return is_valid

    def get_device(self, device_id: str) -> Optional[Dict[str, Any]]:
        return self.devices.get(device_id)

    def list_authorized_devices(self) -> List[Dict[str, Any]]:
        return [d for d in self.devices.values() if d["status"] == "AUTHORIZED"]

    def is_authorized(self, device_id: str) -> bool:
        dev = self.devices.get(device_id)
        return dev is not None and dev["status"] == "AUTHORIZED"

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
        data = {
            "counter": self.counter,
            "devices": self.devices
        }
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self) -> None:
        with open(self.registry_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.counter = data.get("counter", len(data.get("devices", {})))
        self.devices = data.get("devices", {})
