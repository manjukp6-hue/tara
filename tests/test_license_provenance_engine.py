"""
tests/test_license_provenance_engine.py

Verification suite for TARA AI Open-Source License & Code Provenance Engine.
"""

import unittest
import os
import sys
import tempfile
import shutil

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))

from tara_core.license_provenance_engine import (
    LicenseProvenanceEngine,
    UsageStatus,
    ReviewStatus
)


class TestLicenseProvenanceEngine(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.engine = LicenseProvenanceEngine(storage_root=self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_license_classification(self):
        # MIT detection
        mit_text = """
        MIT License
        Copyright (c) 2026 Test Contributors
        Permission is hereby granted, free of charge, to any person obtaining a copy...
        """
        lic, conf, status = self.engine.detect_license_from_text(mit_text)
        self.assertEqual(lic, "MIT")
        self.assertGreater(conf, 0.9)
        self.assertEqual(status, ReviewStatus.VERIFIED_COMPLIANT)

        # Apache 2.0 detection
        apache_text = """
        Licensed under the Apache License, Version 2.0 (the "License");
        http://www.apache.org/licenses/LICENSE-2.0
        """
        lic, conf, status = self.engine.detect_license_from_text(apache_text)
        self.assertEqual(lic, "Apache-2.0")
        self.assertEqual(status, ReviewStatus.REQUIRES_ATTRIBUTION)

        # Copyleft conflict detection
        gpl_text = """
        GNU GENERAL PUBLIC LICENSE, Version 3
        """
        lic, conf, status = self.engine.detect_license_from_text(gpl_text)
        self.assertEqual(lic, "GPL-3.0-or-later")
        self.assertEqual(status, ReviewStatus.FLAGGED_COPYLEFT_CONFLICT)

    def test_02_register_component_and_archive(self):
        sample_file = os.path.join(self.temp_dir, "sample_code.py")
        with open(sample_file, "w") as f:
            f.write("# Hello World open source implementation")

        lic_text = "MIT License\nCopyright (c) 2026 OpenSourceAuthor"
        rec = self.engine.register_imported_component(
            component_name="external_tensor_math",
            repository_url="https://github.com/example/tensor-math",
            version_or_commit="v1.2.0",
            license_text=lic_text,
            file_mappings=[{"source_file": "src/math.py", "destination_file": sample_file}],
            notice_text="Attribution required for OpenSourceAuthor"
        )

        self.assertIsNotNone(rec)
        self.assertEqual(rec.detected_license, "MIT")
        self.assertEqual(rec.usage_status, UsageStatus.ACTIVE)
        self.assertTrue(os.path.exists(os.path.join(self.engine.licenses_dir, f"{rec.record_id}_LICENSE.txt")))

        # Check compliance report
        report = self.engine.generate_compliance_report()
        self.assertEqual(report["total_registered_components"], 1)
        self.assertEqual(report["compliance_status"], "COMPLIANT")

    def test_03_removal_verification_lifecycle(self):
        dummy_file = os.path.join(self.temp_dir, "dummy_mod.py")
        with open(dummy_file, "w") as f:
            f.write("test_code")

        rec = self.engine.register_imported_component(
            component_name="temp_component",
            repository_url="https://github.com/example/temp",
            version_or_commit="1.0",
            license_text="MIT License",
            file_mappings=[{"source_file": "temp.py", "destination_file": dummy_file}]
        )

        # Try to retire before deleting file
        res = self.engine.verify_and_record_removal(rec.record_id)
        self.assertEqual(res["status"], "CANNOT_RETIRE")

        # Now remove file and verify retirement
        os.remove(dummy_file)
        res_ok = self.engine.verify_and_record_removal(rec.record_id)
        self.assertEqual(res_ok["status"], "SUCCESS")
        self.assertEqual(res_ok["usage_status"], UsageStatus.REMOVED.value)

    def test_04_independent_rewrite_verification(self):
        target_file = os.path.join(self.temp_dir, "algorithm.py")
        rec = self.engine.register_imported_component(
            component_name="orig_algo",
            repository_url="https://github.com/example/algo",
            version_or_commit="commit_abc123",
            license_text="MIT License\nCopyright (c) 2024 Jane Doe",
            file_mappings=[{"source_file": "algo.py", "destination_file": target_file}]
        )

        # 1. Code retaining original author marker -> attribution required
        res1 = self.engine.verify_independent_rewrite(rec.record_id, target_file, "def fn():\n    # Author: Jane Doe\n    pass")
        self.assertFalse(res1["is_fully_rewritten"])

        # 2. Independent rewrite without original author markers
        res2 = self.engine.verify_independent_rewrite(rec.record_id, target_file, "def custom_tara_math(x):\n    return x * 42")
        self.assertTrue(res2["is_fully_rewritten"])
        self.assertEqual(res2["usage_status"], UsageStatus.FULLY_REWRITTEN.value)


if __name__ == "__main__":
    unittest.main()
