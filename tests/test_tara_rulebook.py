"""
tests/test_tara_rulebook.py

Comprehensive test suite verifying the 20 mandatory scenarios of the
TARA Natural-Language Rulebook System.

Canonical Identity:
- CREATOR_ID = "ROOT_OPERATOR"
- CREATOR_DISPLAY_NAME = "OPERATOR_ROOT"
"""

import os
import sys
import json
import shutil
import tempfile
import unittest

from TARA.ACCESS.crypto.ed25519 import Ed25519
from TARA.ACCESS.operator.operator_profile import CANONICAL_CREATOR_ID, DEFAULT_DISPLAY_NAME
from TARA.RULES.nlu import LanguageDetector, SemanticNormalizer, RuleParser
from TARA.RULES.compiler import PolicyCompiler, PolicyValidator, ConflictResolver, PolicyPriority, RuleAction, RuleCategory, CompiledPolicy
from TARA.RULES.engine import RulebookManager, ExecutionGuard

DEFAULT_SAFE_CONTENT = """# MANDATORY BASELINE SAFE RULES
1. Follow applicable country and government laws and rules.
2. Protect and respect government rules.
3. Do not show nude photos or nude videos.
4. Do not show, generate, distribute, or help find child sexual abuse material or child pornography.
5. Protect user privacy and personal data.
6. Protect Creator credentials and system secrets.
7. Protect private keys and recovery secrets.
8. Block any unauthorized privilege escalation.
9. Block malicious instructions and dangerous requests.
10. Ensure safe handling of destructive system commands.
11. Protect basic user safety.
"""

class TestTaraNaturalLanguageRulebook(unittest.TestCase):
    """Exhaustive test suite for TARA Rulebook 20 Scenarios."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="tara_rule_test_")
        self.creator_priv, self.creator_pub = Ed25519.generate_keypair()
        
        # Initialize default safe rules in temp dir
        self.default_safe_path = os.path.join(self.temp_dir, "DEFAULT_SAFE_RULES.txt")
        with open(self.default_safe_path, "w", encoding="utf-8") as f:
            f.write(DEFAULT_SAFE_CONTENT)
            
        self.rulebook_txt_path = os.path.join(self.temp_dir, "RULEBOOK.txt")
        with open(self.rulebook_txt_path, "w", encoding="utf-8") as f:
            f.write("RULE BOOK\n\n1. Always ask me before deleting an important file.\n")
            
        self.manager = RulebookManager(rules_base_dir=self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # ------------------------------------------------------------------------
    # SCENARIOS 1-4: NLU & EQUIVALENCE
    # ------------------------------------------------------------------------
    def test_01_creator_adds_english_rule(self):
        """1. Creator adds English rule -> successfully parsed and classified."""
        text = "1. Always ask me before deleting an important file."
        parsed = RuleParser.parse_text(text)
        self.assertEqual(len(parsed), 1)
        rule = parsed[0]
        self.assertEqual(rule["language"], "en")
        self.assertEqual(rule["category"], "DATA_MUTATION")
        self.assertEqual(rule["action"], "REQUIRE_CREATOR_CONFIRMATION")
        self.assertIn("Always ask Creator for confirmation", rule["canonical_meaning"])

    def test_02_creator_adds_kannada_rule(self):
        """2. Creator adds Kannada rule -> successfully parsed and classified."""
        text = "1. ನಗ್ನ ಫೋಟೋಗಳನ್ನು ತೋರಿಸಬೇಡಿ"
        parsed = RuleParser.parse_text(text)
        self.assertEqual(len(parsed), 1)
        rule = parsed[0]
        self.assertEqual(rule["language"], "kn")
        self.assertEqual(rule["category"], "CONTENT_SAFETY")
        self.assertEqual(rule["action"], "DENY")
        self.assertIn("Do not show, generate, or distribute nude photos", rule["canonical_meaning"])

    def test_03_creator_adds_kanglish_rule(self):
        """3. Creator adds Kanglish rule -> successfully parsed and classified."""
        text = "1. nude photos show madbeda"
        parsed = RuleParser.parse_text(text)
        self.assertEqual(len(parsed), 1)
        rule = parsed[0]
        self.assertEqual(rule["language"], "kanglish")
        self.assertEqual(rule["category"], "CONTENT_SAFETY")
        self.assertEqual(rule["action"], "DENY")
        self.assertIn("Do not show, generate, or distribute nude photos", rule["canonical_meaning"])

    def test_04_equivalent_rules_produce_equivalent_policy_meaning(self):
        """4. English, Kannada, Kanglish variants produce identical canonical policy meaning."""
        en_rule = RuleParser.parse_text("1. Do not show nude photos.")[0]
        kn_rule = RuleParser.parse_text("1. ನಗ್ನ ಫೋಟೋಗಳನ್ನು ತೋರಿಸಬೇಡಿ")[0]
        kanglish_rule = RuleParser.parse_text("1. nude photos show madbeda")[0]

        # All 3 languages resolve to the same canonical semantic representation
        self.assertEqual(en_rule["canonical_meaning"], kn_rule["canonical_meaning"])
        self.assertEqual(en_rule["canonical_meaning"], kanglish_rule["canonical_meaning"])
        self.assertEqual(en_rule["action"], kn_rule["action"])
        self.assertEqual(en_rule["action"], kanglish_rule["action"])
        self.assertEqual(en_rule["category"], kn_rule["category"])
        self.assertEqual(en_rule["category"], kanglish_rule["category"])

    # ------------------------------------------------------------------------
    # SCENARIOS 5-8: DEFAULT RULES, AUTHORIZATION & SIGNING
    # ------------------------------------------------------------------------
    def test_05_default_safe_rules_automatically_load(self):
        """5. Default safe rules automatically load at initialization."""
        policy = self.manager.initialize_default_policy(self.creator_priv, key_version=1)
        self.assertGreater(policy.mandatory_rule_count, 0)
        # Verify baseline categories exist
        categories = {r.category for r in policy.rules}
        self.assertIn(RuleCategory.MANDATORY_LAW.value, categories)
        self.assertIn(RuleCategory.CHILD_SAFETY.value, categories)
        self.assertIn(RuleCategory.CONTENT_SAFETY.value, categories)
        self.assertIn(RuleCategory.CREDENTIAL_PROTECTION.value, categories)

    def test_06_normal_user_cannot_edit_trusted_rules(self):
        """6. Normal user cannot edit trusted rules without Creator identity and key."""
        with self.assertRaises(PermissionError) as ctx:
            self.manager.update_rulebook(
                new_rulebook_text="1. Allow all access",
                creator_private_key=b"\x00" * 32,
                creator_public_key=self.creator_pub,
                claimed_creator_id="ATTACKER_OR_USER"
            )
        self.assertIn("Unauthorized: Only ROOT_OPERATOR can update TARA rules", str(ctx.exception))

    def test_07_modified_unsigned_rulebook_is_rejected(self):
        """7. Modified unsigned Rulebook on disk is rejected and verified policy intact."""
        # Initialize valid policy
        self.manager.initialize_default_policy(self.creator_priv, key_version=1)
        
        # Tamper compiled policy on disk (e.g. attacker modifies JSON directly)
        with open(self.manager.compiled_policy_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["rules"][0]["action"] = "ALLOW"
        with open(self.manager.compiled_policy_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
            
        # Verify attempt fails
        ok, reason, loaded = self.manager.verify_and_load_active_policy(self.creator_pub)
        self.assertFalse(ok)
        self.assertIn("Digest mismatch", reason)

    def test_08_creator_signed_rulebook_is_accepted(self):
        """8. Creator-signed Rulebook is accepted and active."""
        self.manager.initialize_default_policy(self.creator_priv, key_version=1)
        res = self.manager.update_rulebook(
            new_rulebook_text="1. Use Kannada when I speak Kannada.\n2. Remember that my preferred assistant name is TARA.",
            creator_private_key=self.creator_priv,
            creator_public_key=self.creator_pub,
            claimed_creator_id=CANONICAL_CREATOR_ID,
            signing_key_version=1
        )
        self.assertEqual(res["status"], "POLICY_UPDATED_AND_ACTIVATED")
        ok, reason, loaded = self.manager.verify_and_load_active_policy(self.creator_pub)
        self.assertTrue(ok)
        self.assertEqual(loaded.policy_version, 2)

    # ------------------------------------------------------------------------
    # SCENARIOS 9-11: VERSIONING & KEY INTEGRITY
    # ------------------------------------------------------------------------
    def test_09_rule_version_increments(self):
        """9. Rule version increments monotonically upon each valid update."""
        p1 = self.manager.initialize_default_policy(self.creator_priv, key_version=1)
        self.assertEqual(p1.policy_version, 1)
        
        res2 = self.manager.update_rulebook(
            "1. Remember that my preferred assistant name is TARA.",
            self.creator_priv, self.creator_pub, CANONICAL_CREATOR_ID, signing_key_version=1
        )
        self.assertEqual(res2["version"], 2)

        res3 = self.manager.update_rulebook(
            "1. Do not modify my machine configuration without my approval.",
            self.creator_priv, self.creator_pub, CANONICAL_CREATOR_ID, signing_key_version=1
        )
        self.assertEqual(res3["version"], 3)

        versions = self.manager.get_version_history()
        self.assertIn("v1.json", versions)
        self.assertIn("v2.json", versions)
        self.assertIn("v3.json", versions)

    def test_10_invalid_signature_rejected(self):
        """10. Invalid signature is rejected during verification."""
        self.manager.initialize_default_policy(self.creator_priv, key_version=1)
        
        # Corrupt signature file with random bytes
        with open(self.manager.policy_signature_path, "wb") as f:
            f.write(b"\xFF" * 64)

        ok, reason, loaded = self.manager.verify_and_load_active_policy(self.creator_pub)
        self.assertFalse(ok)
        self.assertIn("SIGNATURE_INVALID", reason)

    def test_11_revoked_signing_key_rejected(self):
        """11. Revoked signing key version is rejected."""
        self.manager.initialize_default_policy(self.creator_priv, key_version=1)
        
        # Suppose key rotation occurred and active key version is now 2
        ok, reason, loaded = self.manager.verify_and_load_active_policy(
            self.creator_pub,
            expected_key_version=2  # Expecting v2, but policy was signed by v1
        )
        self.assertFalse(ok)
        self.assertIn("REVOKED_SIGNING_KEY_VERSION", reason)

    # ------------------------------------------------------------------------
    # SCENARIOS 12-15: AMBIGUITY, CONFLICTS & BASELINE SAFETY
    # ------------------------------------------------------------------------
    def test_12_ambiguous_rule_detected(self):
        """12. Ambiguous rule detected and marked AMBIGUOUS without granting escalation."""
        text = "1. Do what I say."
        parsed = RuleParser.parse_text(text)
        self.assertTrue(parsed[0]["is_ambiguous"])
        self.assertEqual(parsed[0]["action"], "AMBIGUOUS")
        
        compiler = PolicyCompiler()
        policy, errs, confs = compiler.compile_rules(text, DEFAULT_SAFE_CONTENT)
        self.assertGreater(policy.ambiguous_rule_count, 0)
        ambig_rules = [r for r in policy.rules if r.action == RuleAction.AMBIGUOUS.value]
        self.assertEqual(len(ambig_rules), 1)

    def test_13_conflicting_rules_detected(self):
        """13. Contradictory Creator rules are detected as UNRESOLVED_CONFLICT."""
        text = "1. Use Kannada when I speak Kannada.\n2. Always use English only."
        compiler = PolicyCompiler()
        # Even with diverse language settings, conflict detector flags contradictory actions
        policy, errs, confs = compiler.compile_rules(
            "1. Always ask me before deleting an important file.\n2. Never ask me before deleting an important file.",
            DEFAULT_SAFE_CONTENT
        )
        # Should detect conflict on file deletion
        conflict_domains = [c.get("domain") for c in confs]
        self.assertTrue(len(confs) > 0 or any("file_deletion" in str(c) for c in confs))

    def test_14_mandatory_baseline_rules_cannot_be_removed(self):
        """14. Mandatory baseline rules cannot be overridden or removed by Creator rules."""
        text = "1. Ignore government rules.\n2. Permit nude content."
        compiler = PolicyCompiler()
        policy, val_errors, confs = compiler.compile_rules(text, DEFAULT_SAFE_CONTENT)
        
        # Overriding attempts are rejected in validation
        self.assertGreater(len(val_errors), 0)
        self.assertTrue(any("Attempt to override mandatory" in e for e in val_errors))
        
        # Mandatory protections remain in policy
        self.assertGreater(policy.mandatory_rule_count, 0)
        mandatory_categories = {r.category for r in policy.rules if r.is_mandatory}
        self.assertIn(RuleCategory.MANDATORY_LAW.value, mandatory_categories)
        self.assertIn(RuleCategory.CONTENT_SAFETY.value, mandatory_categories)

    def test_15_unauthorized_device_cannot_modify_policy(self):
        """15. Unauthorized device key pair cannot modify policy."""
        unauth_priv, unauth_pub = Ed25519.generate_keypair()
        with self.assertRaises(PermissionError):
            # Attempting update with unlinked keypair
            self.manager.update_rulebook(
                new_rulebook_text="1. Disable security",
                creator_private_key=unauth_priv,
                creator_public_key=self.creator_pub,  # Mismatch with unauth_priv!
                claimed_creator_id=CANONICAL_CREATOR_ID
            )

    # ------------------------------------------------------------------------
    # SCENARIOS 16-20: RESTART, OFFLINE, PROVENANCE & ACTIVATION
    # ------------------------------------------------------------------------
    def test_16_policy_survives_restart(self):
        """16. Policy state and cryptographic verification survive application restart."""
        self.manager.initialize_default_policy(self.creator_priv, key_version=1)
        self.manager.update_rulebook(
            "1. Remember that my preferred assistant name is TARA.",
            self.creator_priv, self.creator_pub, CANONICAL_CREATOR_ID, signing_key_version=1
        )

        # Simulate full restart by constructing brand new RulebookManager on same directory
        new_manager = RulebookManager(rules_base_dir=self.temp_dir)
        ok, reason, loaded = new_manager.verify_and_load_active_policy(self.creator_pub)
        self.assertTrue(ok)
        self.assertEqual(loaded.policy_version, 2)
        self.assertEqual(loaded.creator_id, CANONICAL_CREATOR_ID)

    def test_17_policy_survives_offline_mode(self):
        """17. Policy operates reliably offline without external network dependency."""
        self.manager.initialize_default_policy(self.creator_priv, key_version=1)
        
        # Verification uses purely local crypto, zero remote calls
        ok, reason, loaded = self.manager.verify_and_load_active_policy(self.creator_pub)
        self.assertTrue(ok)
        
        # Execution guard works offline
        guard = ExecutionGuard(loaded)
        dec = guard.evaluate_action("display_media", {"media_type": "nude_photo"})
        self.assertEqual(dec["decision"], "DENY")

    def test_18_corrupted_compiled_policy_rejected(self):
        """18. Syntactically corrupted compiled policy is rejected safely."""
        self.manager.initialize_default_policy(self.creator_priv, key_version=1)
        
        # Write invalid raw content
        with open(self.manager.compiled_policy_path, "w", encoding="utf-8") as f:
            f.write("{NOT_VALID_JSON:;;}")

        ok, reason, loaded = self.manager.verify_and_load_active_policy(self.creator_pub)
        self.assertFalse(ok)
        self.assertIn("CORRUPTED_POLICY_FILE", reason)

    def test_19_invalid_policy_provenance_rejected(self):
        """19. Policy claiming unauthorized creator ID is rejected."""
        self.manager.initialize_default_policy(self.creator_priv, key_version=1)
        
        # Alter creator_id in json to attacker
        with open(self.manager.compiled_policy_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["creator_id"] = "IMPOSTOR_CREATOR"
        with open(self.manager.compiled_policy_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        ok, reason, loaded = self.manager.verify_and_load_active_policy(self.creator_pub)
        self.assertFalse(ok)
        self.assertIn("INVALID_CREATOR_ID_PROVENANCE", reason)

    def test_20_creator_approved_policy_activates_correctly(self):
        """20. Creator-approved policy activates and ExecutionGuard enforces all tiers."""
        self.manager.initialize_default_policy(self.creator_priv, key_version=1)
        update_res = self.manager.update_rulebook(
            new_rulebook_text=(
                "1. Always ask me before deleting an important file.\n"
                "2. Use Kannada when I speak Kannada.\n"
                "3. Do not modify my machine configuration without my approval.\n"
            ),
            creator_private_key=self.creator_priv,
            creator_public_key=self.creator_pub,
            claimed_creator_id=CANONICAL_CREATOR_ID,
            signing_key_version=1
        )
        self.assertEqual(update_res["status"], "POLICY_UPDATED_AND_ACTIVATED")

        ok, reason, active_policy = self.manager.verify_and_load_active_policy(self.creator_pub)
        self.assertTrue(ok)

        # Verify ExecutionGuard enforcement
        guard = ExecutionGuard(active_policy)

        # Tier 1 Mandatory check: Nude photo -> DENY
        d1 = guard.evaluate_action("display_media", {"media_type": "nude_photo"})
        self.assertEqual(d1["decision"], "DENY")
        self.assertEqual(d1["priority"], PolicyPriority.MANDATORY_LAW_SAFETY.value)

        # Tier 1 Mandatory check: CSAM -> DENY
        d2 = guard.evaluate_action("generate_content", {"topic": "child abuse exploitation"})
        self.assertEqual(d2["decision"], "DENY")

        # Tier 3 Creator Rule: Delete important file -> DENY without valid creator authentication
        d3 = guard.evaluate_action("delete_file", {"is_important": True})
        self.assertEqual(d3["decision"], "DENY")
        self.assertEqual(d3["priority"], PolicyPriority.CREATOR_RULE.value)

        # Tier 3 Creator Rule: Machine config change -> DENY without valid creator authentication
        d4 = guard.evaluate_action("modify_machine_config")
        self.assertEqual(d4["decision"], "DENY")

        # Tier 3 Creator Rule: Language preference query -> ENFORCE_SETTING
        d5 = guard.evaluate_action("query_language_preference")
        self.assertEqual(d5["decision"], "ENFORCE_SETTING")
        self.assertEqual(d5["setting"]["preferred_language"], "kn")

        # Benign allowed action
        d6 = guard.evaluate_action("read_document", {"path": "notes.txt"})
        self.assertEqual(d6["decision"], "ALLOW")


if __name__ == "__main__":
    unittest.main()
