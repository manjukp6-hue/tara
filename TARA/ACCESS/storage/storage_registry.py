"""
TARA/ACCESS/storage/storage_registry.py

Centralized Storage Registry for TARA.
Registers and tracks all persistent storage locations (local, external, cloud).
Enforces common lifecycle:
  register_storage() -> list_tara_objects() -> delete_tara_objects() -> verify_deletion() -> finalize_destroy()

Safety Rules:
- Delete ONLY data identified as TARA-owned.
- Never delete unrelated personal files, photos, documents, OS files, applications.
- If ownership is ambiguous: DO NOT DELETE. Report DELETION_REQUIRES_REVIEW.
- If remote deletion cannot be verified: Report DELETION_NOT_VERIFIABLE.
"""

import os
import shutil
import secrets
from enum import Enum
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Set


class StorageType(str, Enum):
    INTERNAL = "INTERNAL"
    SECONDARY = "SECONDARY"
    EXTERNAL_SSD = "EXTERNAL_SSD"
    USB = "USB"
    SD_CARD = "SD_CARD"
    NAS = "NAS"
    GOOGLE_DRIVE = "GOOGLE_DRIVE"
    CLOUD_STORAGE = "CLOUD_STORAGE"
    CUSTOM = "CUSTOM"


class OwnershipTag(str, Enum):
    TARA_OWNED = "TARA_OWNED"
    AMBIGUOUS = "AMBIGUOUS"
    USER_CONTENT = "USER_CONTENT"


class DeletionStatus(str, Enum):
    SUCCESS = "SUCCESS"
    DELETION_NOT_VERIFIABLE = "DELETION_NOT_VERIFIABLE"
    DELETION_REQUIRES_REVIEW = "DELETION_REQUIRES_REVIEW"
    PARTIAL_FAILURE = "PARTIAL_FAILURE"


class StorageObject:
    def __init__(
        self,
        uri: str,
        name: str,
        ownership: OwnershipTag,
        size_bytes: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.uri = uri
        self.name = name
        self.ownership = ownership
        self.size_bytes = size_bytes
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uri": self.uri,
            "name": self.name,
            "ownership": self.ownership.value,
            "size_bytes": self.size_bytes,
            "metadata": self.metadata
        }


class DeletionReport:
    def __init__(
        self,
        location_id: str,
        status: DeletionStatus,
        deleted_count: int = 0,
        verified_count: int = 0,
        ambiguous_count: int = 0,
        details: Optional[str] = None
    ):
        self.location_id = location_id
        self.status = status
        self.deleted_count = deleted_count
        self.verified_count = verified_count
        self.ambiguous_count = ambiguous_count
        self.details = details or ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "location_id": self.location_id,
            "status": self.status.value,
            "deleted_count": self.deleted_count,
            "verified_count": self.verified_count,
            "ambiguous_count": self.ambiguous_count,
            "details": self.details
        }


class StorageProvider(ABC):
    def __init__(self, location_id: str, storage_type: StorageType, base_uri: str, is_external: bool = False):
        self.location_id = location_id
        self.storage_type = storage_type
        self.base_uri = base_uri
        self.is_external = is_external

    @abstractmethod
    def list_tara_objects(self) -> List[StorageObject]:
        pass

    @abstractmethod
    def delete_tara_objects(self, verify: bool = True) -> DeletionReport:
        pass

    @abstractmethod
    def verify_deletion(self) -> bool:
        pass

    @abstractmethod
    def finalize_destroy(self) -> bool:
        pass


class LocalStorageProvider(StorageProvider):
    """
    Provider for local filesystem locations (internal disk, secondary partitions).
    """
    KNOWN_TARA_SUBDIRECTORIES = {
        "model", "skills", "knowledge", "tools", "memory", "identity",
        "cache", "logs", "config", "temp", "index", "database",
        "datasets", "learning", "episodes"
    }
    PROTECTED_USER_EXTENSIONS = {".jpg", ".jpeg", ".png", ".docx", ".pdf", ".xlsx", ".pptx", ".mp4", ".mp3", ".zip", ".rar", ".7z"}

    def __init__(self, location_id: str, root_path: str, storage_type: StorageType = StorageType.INTERNAL):
        super().__init__(location_id, storage_type, root_path, is_external=False)
        self.root_path = os.path.abspath(root_path)

    def _classify_file(self, full_path: str) -> OwnershipTag:
        rel = os.path.relpath(full_path, self.root_path).lower()
        parts = rel.replace("\\", "/").split("/")
        ext = os.path.splitext(full_path)[1].lower()

        # Check if file has protected personal content extension or unknown archive
        if ext in self.PROTECTED_USER_EXTENSIONS:
            return OwnershipTag.AMBIGUOUS

        # If root directory itself is a dedicated TARA subsystem
        root_name = os.path.basename(self.root_path).lower()
        if root_name in self.KNOWN_TARA_SUBDIRECTORIES:
            return OwnershipTag.TARA_OWNED

        # If any parent directory component matches known TARA subsystems
        for part in parts[:-1]:
            if part in self.KNOWN_TARA_SUBDIRECTORIES:
                return OwnershipTag.TARA_OWNED

        # If file basename explicitly matches known TARA files or prefixes
        basename = parts[-1]
        if any(prefix in basename for prefix in ("tara_", "creator_", "device_", "weights", "skill", "episodes", "manifest")):
            return OwnershipTag.TARA_OWNED

        # Default to AMBIGUOUS to protect unknown user files
        return OwnershipTag.AMBIGUOUS

    def list_tara_objects(self) -> List[StorageObject]:
        objects = []
        if not os.path.exists(self.root_path):
            return objects

        if os.path.isfile(self.root_path):
            tag = self._classify_file(self.root_path)
            objects.append(StorageObject(
                uri=self.root_path,
                name=os.path.basename(self.root_path),
                ownership=tag,
                size_bytes=os.path.getsize(self.root_path)
            ))
            return objects

        for root, dirs, files in os.walk(self.root_path):
            for file in files:
                full_path = os.path.join(root, file)
                tag = self._classify_file(full_path)
                size = os.path.getsize(full_path) if os.path.exists(full_path) else 0
                objects.append(StorageObject(
                    uri=full_path,
                    name=file,
                    ownership=tag,
                    size_bytes=size
                ))
        return objects

    def delete_tara_objects(self, verify: bool = True) -> DeletionReport:
        if not os.path.exists(self.root_path):
            return DeletionReport(self.location_id, DeletionStatus.SUCCESS, 0, 0, 0, "Path does not exist")

        all_objs = self.list_tara_objects()
        deleted = 0
        ambiguous = 0

        for obj in all_objs:
            if obj.ownership == OwnershipTag.AMBIGUOUS:
                ambiguous += 1
                # NEVER delete ambiguous or personal user files!
                continue
            elif obj.ownership == OwnershipTag.TARA_OWNED:
                try:
                    if os.path.isfile(obj.uri):
                        # Overwrite sensitive files with random bytes before unlink
                        try:
                            file_size = os.path.getsize(obj.uri)
                            with open(obj.uri, "wb") as f:
                                f.write(secrets.token_bytes(min(file_size, 2048)))
                        except Exception:
                            pass
                        os.remove(obj.uri)
                        deleted += 1
                except Exception:
                    pass

        # Clean empty directories
        if os.path.isdir(self.root_path):
            for root, dirs, files in os.walk(self.root_path, topdown=False):
                if not files and not dirs:
                    try:
                        os.rmdir(root)
                    except Exception:
                        pass
            if os.path.exists(self.root_path) and not os.listdir(self.root_path):
                try:
                    os.rmdir(self.root_path)
                except Exception:
                    pass

        verified = deleted
        if verify:
            verified = sum(1 for obj in all_objs if obj.ownership == OwnershipTag.TARA_OWNED and not os.path.exists(obj.uri))

        status = DeletionStatus.SUCCESS
        if ambiguous > 0:
            status = DeletionStatus.DELETION_REQUIRES_REVIEW

        return DeletionReport(
            location_id=self.location_id,
            status=status,
            deleted_count=deleted,
            verified_count=verified,
            ambiguous_count=ambiguous,
            details=f"Deleted {deleted} TARA files; {ambiguous} ambiguous files spared."
        )

    def verify_deletion(self) -> bool:
        objs = self.list_tara_objects()
        tara_files = [o for o in objs if o.ownership == OwnershipTag.TARA_OWNED]
        return len(tara_files) == 0

    def finalize_destroy(self) -> bool:
        if os.path.exists(self.root_path):
            try:
                if not os.listdir(self.root_path):
                    shutil.rmtree(self.root_path, ignore_errors=True)
            except Exception:
                pass
        return not os.path.exists(self.root_path)


class ExternalStorageProvider(LocalStorageProvider):
    """Provider for external SSD, USB, SD Card, NAS mounts."""
    def __init__(self, location_id: str, mount_path: str, storage_type: StorageType = StorageType.EXTERNAL_SSD):
        super().__init__(location_id, mount_path, storage_type=storage_type)
        self.is_external = True


class CloudStorageProvider(StorageProvider):
    """
    Provider abstraction for remote cloud buckets (Google Drive, Firebase Cloud Storage, AWS S3).
    """
    def __init__(
        self,
        location_id: str,
        cloud_uri: str,
        storage_type: StorageType = StorageType.CLOUD_STORAGE,
        supports_remote_verification: bool = True
    ):
        super().__init__(location_id, storage_type, cloud_uri, is_external=True)
        self.supports_remote_verification = supports_remote_verification
        # In-memory mock cloud object bucket for testing & runtime abstraction
        self._cloud_objects: Dict[str, Dict[str, Any]] = {}

    def register_cloud_object(self, key: str, data: bytes, ownership: OwnershipTag = OwnershipTag.TARA_OWNED) -> None:
        self._cloud_objects[key] = {"data": data, "ownership": ownership, "size": len(data)}

    def list_tara_objects(self) -> List[StorageObject]:
        return [
            StorageObject(
                uri=f"{self.base_uri}/{k}",
                name=k,
                ownership=v["ownership"],
                size_bytes=v["size"]
            )
            for k, v in self._cloud_objects.items()
        ]

    def delete_tara_objects(self, verify: bool = True) -> DeletionReport:
        all_objs = list(self._cloud_objects.items())
        deleted = 0
        ambiguous = 0

        for k, v in all_objs:
            if v["ownership"] == OwnershipTag.AMBIGUOUS:
                ambiguous += 1
            elif v["ownership"] == OwnershipTag.TARA_OWNED:
                del self._cloud_objects[k]
                deleted += 1

        if not self.supports_remote_verification:
            return DeletionReport(
                location_id=self.location_id,
                status=DeletionStatus.DELETION_NOT_VERIFIABLE,
                deleted_count=deleted,
                verified_count=0,
                ambiguous_count=ambiguous,
                details="Cloud provider API does not support cryptographic deletion receipt or verification."
            )

        status = DeletionStatus.SUCCESS if ambiguous == 0 else DeletionStatus.DELETION_REQUIRES_REVIEW
        return DeletionReport(
            location_id=self.location_id,
            status=status,
            deleted_count=deleted,
            verified_count=deleted,
            ambiguous_count=ambiguous,
            details=f"Cloud deletion completed for {deleted} objects."
        )

    def verify_deletion(self) -> bool:
        if not self.supports_remote_verification:
            return False
        return len([k for k, v in self._cloud_objects.items() if v["ownership"] == OwnershipTag.TARA_OWNED]) == 0

    def finalize_destroy(self) -> bool:
        self._cloud_objects.clear()
        return True


class TaraStorageRegistry:
    """
    Centralized registry of all persistent TARA storage locations.
    Enables future storage types and coordinated deletion across all providers.
    """
    def __init__(self, base_repo_dir: Optional[str] = None):
        if base_repo_dir is None:
            base_repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
        self.base_repo_dir = base_repo_dir
        self.providers: Dict[str, StorageProvider] = {}
        self._register_default_providers()

    def _register_default_providers(self) -> None:
        """Registers default TARA directories."""
        default_dirs = [
            ("local_model", os.path.join(self.base_repo_dir, "TARA", "MODEL")),
            ("local_skills", os.path.join(self.base_repo_dir, "TARA", "SKILLS")),
            ("local_knowledge", os.path.join(self.base_repo_dir, "TARA", "KNOWLEDGE")),
            ("local_tools", os.path.join(self.base_repo_dir, "TARA", "TOOLS")),
            ("local_memory", os.path.join(self.base_repo_dir, "TARA", "MEMORY")),
            ("local_identity", os.path.join(self.base_repo_dir, "TARA", "ACCESS")),
            ("local_storage", os.path.join(self.base_repo_dir, "storage")),
        ]
        for loc_id, p in default_dirs:
            self.register_provider(loc_id, LocalStorageProvider(loc_id, p, StorageType.INTERNAL))

    def register_provider(self, location_id: str, provider: StorageProvider) -> None:
        self.providers[location_id] = provider

    def unregister_provider(self, location_id: str) -> None:
        self.providers.pop(location_id, None)

    def enumerate_all_objects(self) -> Dict[str, List[Dict[str, Any]]]:
        result = {}
        for loc_id, prov in self.providers.items():
            result[loc_id] = [o.to_dict() for o in prov.list_tara_objects()]
        return result

    def delete_all_tara_storage(self, verify: bool = True) -> Dict[str, DeletionReport]:
        reports = {}
        for loc_id, prov in list(self.providers.items()):
            reports[loc_id] = prov.delete_tara_objects(verify=verify)
        return reports

    def verify_all_deleted(self) -> bool:
        return all(prov.verify_deletion() for prov in self.providers.values())

    def finalize_destroy_all(self) -> bool:
        all_ok = True
        for prov in list(self.providers.values()):
            if not prov.finalize_destroy():
                all_ok = False
        self.providers.clear()
        return all_ok
