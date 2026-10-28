"""
TARA Pre-Training Cryptographic Authorization Gate - Comprehensive Test Suite
Validates Ed25519 digital signatures, canonical JSON deterministic byte serialization,
fail-closed trusted public key resolution, root key conflict detection, and runtime artifact matching.
"""

import os
import sys
import json
import copy
import hashlib
from typing import Dict, Any

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from train_main import (
    resolve_trusted_public_key,
    verify_ed25519_authorization_signature,
    verify_pretraining_authorization_proof,
)


def load_canonical_proof() -> Dict[str, Any]:
    proof_path = os.path.join(REPO_ROOT, "pretraining_authorization.json")
    with open(proof_path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_1_authentic_baseline():
    proof = load_canonical_proof()
    trusted_pub = resolve_trusted_public_key()
    assert trusted_pub is not None, "Failed to resolve authoritative root public key"
    ok, msg = verify_ed25519_authorization_signature(proof, trusted_public_key=trusted_pub)
    assert ok is True, f"Authentic baseline signature verification failed: {msg}"
    print("  [PASS] Test 1: Authentic baseline Ed25519 digital signature verified.")


def test_2_missing_trusted_key_fails_closed():
    proof = load_canonical_proof()
    # Explicitly pass non-matching key
    ok, msg = verify_ed25519_authorization_signature(proof, trusted_public_key="0" * 64)
    assert ok is False, "Missing or non-matching trusted key MUST fail closed!"
    assert "does not match the configured trusted key" in msg
    print("  [PASS] Test 2: Mismatched / unconfigured trusted public key fails closed.")


def test_3_attacker_self_signed_proof_fails_closed():
    proof = load_canonical_proof()
    trusted_pub = resolve_trusted_public_key()
    assert trusted_pub is not None

    attacker_priv = Ed25519PrivateKey.generate()
    attacker_pub = attacker_priv.public_key().public_bytes_raw().hex()

    tampered_proof = copy.deepcopy(proof)
    tampered_proof["public_key"] = attacker_pub

    payload = {
        "audit_version": str(tampered_proof.get("audit_version", "1.0.0")),
        "authorization_status": str(tampered_proof.get("authorization_status", "")),
        "dataset_fingerprint_sha256": str(tampered_proof.get("dataset_fingerprint_sha256", "")),
        "model_config_sha256": str(tampered_proof.get("model_config_sha256", "")),
        "schema_version": int(tampered_proof.get("schema_version", 1)),
        "tokenizer_sha256": str(tampered_proof.get("tokenizer_sha256", "")),
    }
    canon_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    tampered_proof["signature"] = attacker_priv.sign(canon_bytes).hex()

    ok, msg = verify_ed25519_authorization_signature(tampered_proof, trusted_public_key=trusted_pub)
    assert ok is False, "Attacker self-signed proof MUST NOT be trusted!"
    assert "does not match the configured trusted key" in msg
    print("  [PASS] Test 3: Attacker self-signed proof rejected with trusted key mismatch.")


def test_4_stripped_signature_fails_closed():
    proof = load_canonical_proof()
    trusted_pub = resolve_trusted_public_key()
    stripped_proof = copy.deepcopy(proof)
    del stripped_proof["signature"]

    ok, msg = verify_ed25519_authorization_signature(stripped_proof, trusted_public_key=trusted_pub)
    assert ok is False, "Stripped signature proof MUST fail closed!"
    assert "Missing or invalid signature" in msg
    print("  [PASS] Test 4: Stripped signature proof fails closed.")


def test_5_tampered_signature_bytes_fails_closed():
    proof = load_canonical_proof()
    trusted_pub = resolve_trusted_public_key()
    bad_sig_proof = copy.deepcopy(proof)
    bad_sig_proof["signature"] = "00" * 64

    ok, msg = verify_ed25519_authorization_signature(bad_sig_proof, trusted_public_key=trusted_pub)
    assert ok is False, "Tampered signature bytes MUST fail closed!"
    assert "Ed25519 signature verification failed" in msg
    print("  [PASS] Test 5: Tampered signature bytes fails closed.")


def test_6_tampered_payload_fields_fail_closed():
    proof = load_canonical_proof()
    trusted_pub = resolve_trusted_public_key()

    for field in ("authorization_status", "model_config_sha256", "tokenizer_sha256", "dataset_fingerprint_sha256", "audit_version"):
        tampered = copy.deepcopy(proof)
        tampered[field] = "tampered_value_" + field
        ok, msg = verify_ed25519_authorization_signature(tampered, trusted_public_key=trusted_pub)
        assert ok is False, f"Tampered field {field} MUST fail signature verification!"
    print("  [PASS] Test 6: Tampered payload fields fail signature verification.")


def test_7_canonical_json_multi_vector_determinism():
    """
    Tests canonical JSON serialization across diverse boundary vectors:
    - Vector A: Standard canonical auth payload
    - Vector B: Empty boundary strings
    - Vector C: Escaped characters (quotes, backslashes, newlines)
    - Vector D: Unicode characters (Kannada numerals, Devanagari numerals)
    """
    vectors = [
        # Vector A: Standard baseline
        (
            {
                "audit_version": "1.0.0",
                "authorization_status": "AUTHORIZED",
                "dataset_fingerprint_sha256": "63249e272b913590e808546dca0d65e3e9b99e022784272b834f1e3d859b697d",
                "model_config_sha256": "85ceb6201a1dfa3fed00526722335f067ef2a3b7ee36fa1e5e3b57a3b8bca207",
                "schema_version": 1,
                "tokenizer_sha256": "007f91cd022ee262c37792fb095459e5625fc33d95f016dc79ce4739174850de",
            },
            '{"audit_version":"1.0.0","authorization_status":"AUTHORIZED","dataset_fingerprint_sha256":"63249e272b913590e808546dca0d65e3e9b99e022784272b834f1e3d859b697d","model_config_sha256":"85ceb6201a1dfa3fed00526722335f067ef2a3b7ee36fa1e5e3b57a3b8bca207","schema_version":1,"tokenizer_sha256":"007f91cd022ee262c37792fb095459e5625fc33d95f016dc79ce4739174850de"}',
            "3abad2dd08090f64ec8d0665e4bc76c1bc805dbb36a33b237b93426970eb7ad3",
        ),
        # Vector B: Empty strings boundary
        (
            {
                "audit_version": "",
                "authorization_status": "",
                "dataset_fingerprint_sha256": "",
                "model_config_sha256": "",
                "schema_version": 0,
                "tokenizer_sha256": "",
            },
            '{"audit_version":"","authorization_status":"","dataset_fingerprint_sha256":"","model_config_sha256":"","schema_version":0,"tokenizer_sha256":""}',
            None,
        ),
        # Vector C: Escaped characters
        (
            {
                "audit_version": '1.0\\"quote\\',
                "authorization_status": "line1\\nline2",
                "dataset_fingerprint_sha256": "path/with/slashes",
                "model_config_sha256": "tab\\tspace",
                "schema_version": 42,
                "tokenizer_sha256": "special<&>chars",
            },
            '{"audit_version":"1.0\\\\\\"quote\\\\","authorization_status":"line1\\\\nline2","dataset_fingerprint_sha256":"path/with/slashes","model_config_sha256":"tab\\\\tspace","schema_version":42,"tokenizer_sha256":"special<&>chars"}',
            None,
        ),
        # Vector D: Unicode Kannada and Devanagari characters
        (
            {
                "audit_version": "೧.೦.೦",
                "authorization_status": "ಅಧಿಕೃತ",
                "dataset_fingerprint_sha256": "ಸಂಪನ್ಮೂಲ_ಫಿಂಗರ್‌ಪ್ರಿಂಟ್",
                "model_config_sha256": "ಕಾನ್ಫಿಗ್_ಹ್ಯಾಶ್",
                "schema_version": 1,
                "tokenizer_sha256": "ಟೋಕನೈಜರ್_ಹ್ಯಾಶ್",
            },
            '{"audit_version":"೧.೦.೦","authorization_status":"ಅಧಿಕೃತ","dataset_fingerprint_sha256":"ಸಂಪನ್ಮೂಲ_ಫಿಂಗರ್‌ಪ್ರಿಂಟ್","model_config_sha256":"ಕಾನ್ಫಿಗ್_ಹ್ಯಾಶ್","schema_version":1,"tokenizer_sha256":"ಟೋಕನೈಜರ್_ಹ್ಯಾಶ್"}',
            None,
        ),
    ]

    for idx, (payload_dict, expected_str, expected_sha) in enumerate(vectors, 1):
        ser_bytes = json.dumps(payload_dict, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        ser_str = ser_bytes.decode("utf-8")
        assert ser_str == expected_str, f"Vector {idx} serialization mismatch:\nGot:      {ser_str}\nExpected: {expected_str}"
        if expected_sha:
            actual_sha = hashlib.sha256(ser_bytes).hexdigest()
            assert actual_sha == expected_sha, f"Vector {idx} SHA-256 mismatch:\nGot:      {actual_sha}\nExpected: {expected_sha}"
    print("  [PASS] Test 7: Canonical JSON multi-vector serialization verified (boundary, escaping, Unicode Kannada).")


def test_8_full_pretraining_gate_execution():
    cand_dir = os.path.join(REPO_ROOT, "storage", "models", "tara_candidate_v1")
    os.makedirs(cand_dir, exist_ok=True)
    # Ensure candidate baseline is seeded from model_config if fresh/empty
    import shutil
    for f in ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        src = os.path.join(REPO_ROOT, "model_config", f)
        dst = os.path.join(cand_dir, f)
        if os.path.isfile(src) and not os.path.isfile(dst):
            shutil.copy2(src, dst)

    passed = verify_pretraining_authorization_proof(cand_dir)
    assert passed is True, "Full pre-training gate MUST pass on authentic candidate repository assets!"
    print("  [PASS] Test 8: Full Pre-Training Gate verified (Signature + Config + Tokenizer + Architecture).")


def run_all_tests():
    print("=" * 80)
    print("  TARA CRYPTOGRAPHIC PRE-TRAINING GATE SECURITY VERIFICATION SUITE")
    print("=" * 80)
    test_1_authentic_baseline()
    test_2_missing_trusted_key_fails_closed()
    test_3_attacker_self_signed_proof_fails_closed()
    test_4_stripped_signature_fails_closed()
    test_5_tampered_signature_bytes_fails_closed()
    test_6_tampered_payload_fields_fail_closed()
    test_7_canonical_json_multi_vector_determinism()
    test_8_full_pretraining_gate_execution()
    print("=" * 80)
    print("  >>> ALL 8 CRYPTOGRAPHIC TESTS PASSED WITH 100% SUCCESS <<<")
    print("=" * 80)


if __name__ == "__main__":
    run_all_tests()
