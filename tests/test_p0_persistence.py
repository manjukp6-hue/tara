"""
tests/test_p0_persistence.py

Verification & Regression Suite for P0.2:
Render Ephemeral State Loss & Durable Persistence Abstraction.
Ensures:
1. Category A mutable state persists across process restarts.
2. Production without configured durable store fails closed.
3. Concurrent writes with version conflict are handled safely.
4. Corrupted store data fails closed without silent recovery.
5. Canonical creator identity ROOT_OPERATOR is immutable.
6. SQL and Local storage providers operate correctly.
"""

import os
import sys
import json
import shutil
import tempfile
import unittest

sys.path.insert(0, os.path.abspath("python"))

from TARA.ACCESS.storage.durable_storage import (
    DurableStorageManager,
    LocalFileStorageProvider,
    SQLDurableStorageProvider,
    DurableStorageUnconfiguredError,
    StorageCorruptionError,
    ConcurrencyConflictError,
    CANONICAL_CREATOR_ID,
    CANONICAL_DISPLAY_NAME,
    CANONICAL_ROLE,
)


class TestP0Persistence(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        os.environ["TARA_DURABLE_DIR"] = self.temp_dir
        os.environ["TARA_ENV"] = "development"
        os.environ["RENDER"] = "false"
        DurableStorageManager.reset()

    def tearDown(self):
        DurableStorageManager.reset()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_process_restart_persistence(self):
        """Verify data survives simulated process restart."""
        provider1 = LocalFileStorageProvider(self.temp_dir)
        v1 = provider1.set("test:key", b"my_persistent_data")
        self.assertEqual(v1, 1)

        # Drop reference and simulate new process mounting same store
        del provider1
        provider2 = LocalFileStorageProvider(self.temp_dir)
        data = provider2.get("test:key")
        self.assertEqual(data, b"my_persistent_data")

    def test_02_creator_registry_and_record_persistence(self):
        """Verify creator record and registry can be saved and retrieved intact."""
        mgr = DurableStorageManager.get_instance()
        record = {
            "creator_id": CANONICAL_CREATOR_ID,
            "display_name": CANONICAL_DISPLAY_NAME,
            "role": CANONICAL_ROLE,
            "status": "active"
        }
        v = mgr.set_record("creator:record", record)
        self.assertGreater(v, 0)

        retrieved = mgr.get_record("creator:record")
        self.assertEqual(retrieved["creator_id"], CANONICAL_CREATOR_ID)
        self.assertEqual(retrieved["display_name"], CANONICAL_DISPLAY_NAME)
        self.assertEqual(retrieved["role"], CANONICAL_ROLE)

    def test_03_canonical_creator_invariant_protection(self):
        """Ensure attempts to overwrite canonical creator ROOT_OPERATOR are blocked."""
        mgr = DurableStorageManager.get_instance()
        bogus_record = {
            "creator_id": "IMPOSTOR_CREATOR",
            "display_name": "FAKE",
            "role": "ROOT_CREATOR"
        }
        with self.assertRaises(StorageCorruptionError):
            mgr.set_record("creator:record", bogus_record)

    def test_04_concurrency_conflict_detection(self):
        """Verify optimistic concurrency control rejects conflicting versions."""
        provider = LocalFileStorageProvider(self.temp_dir)
        v1 = provider.set("counter", b"1")

        # Second write specifying expected_version=v1 succeeds
        v2 = provider.set("counter", b"2", expected_version=v1)
        self.assertEqual(v2, 2)

        # Stale write specifying old version fails with ConcurrencyConflictError
        with self.assertRaises(ConcurrencyConflictError):
            provider.set("counter", b"3", expected_version=v1)

    def test_05_corrupted_store_fails_closed(self):
        """Verify corrupted storage records fail closed rather than returning invalid data."""
        provider = LocalFileStorageProvider(self.temp_dir)
        provider.set("secure:data", b"authentic_content")

        # Corrupt file bytes directly
        path = provider._get_key_path("secure:data")
        with open(path, "wb") as f:
            f.write(b"CORRUPTED_HEX_HASH_1234567890" + b"tampered_data")

        with self.assertRaises(StorageCorruptionError):
            provider.get("secure:data")

    def test_06_production_unconfigured_fails_closed(self):
        """Verify in production mode without DB, mutations fail closed."""
        os.environ["RENDER"] = "true"
        os.environ["TARA_ENV"] = "production"
        if "TARA_DATABASE_URL" in os.environ:
            del os.environ["TARA_DATABASE_URL"]
        if "DATABASE_URL" in os.environ:
            del os.environ["DATABASE_URL"]
        DurableStorageManager.reset()

        mgr = DurableStorageManager.get_instance()
        self.assertTrue(mgr.is_production_unconfigured())

        with self.assertRaises(DurableStorageUnconfiguredError):
            mgr.set_record("creator:record", {"creator_id": CANONICAL_CREATOR_ID})

    def test_07_sql_storage_provider(self):
        """Verify SQL durable store with SQLite backing."""
        db_path = os.path.join(self.temp_dir, "test_durable.db")
        sql_provider = SQLDurableStorageProvider(f"sqlite:///{db_path}")

        v1 = sql_provider.set("sql:key", b"sql_data")
        self.assertEqual(v1, 1)

        val = sql_provider.get("sql:key")
        self.assertEqual(val, b"sql_data")

        is_healthy, msg = sql_provider.health_check()
        self.assertTrue(is_healthy)

    def test_08_migration_from_local_state(self):
        """Verify migration tool successfully reads and stores existing Category A state."""
        mgr = DurableStorageManager.get_instance()
        results = mgr.migrate_all_local_state()
        self.assertIn("creator:record", results)
        # Verify creator record was migrated
        migrated_creator = mgr.get_record("creator:record")
        self.assertIsNotNone(migrated_creator)
        self.assertEqual(migrated_creator["creator_id"], CANONICAL_CREATOR_ID)


if __name__ == "__main__":
    unittest.main()
