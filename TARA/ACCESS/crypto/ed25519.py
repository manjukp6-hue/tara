"""
TARA/ACCESS/crypto/ed25519.py

Hardened Ed25519 digital signature system (RFC 8032) backed by PyCA cryptography
(OpenSSL C bindings).
Protects against timing attacks, branch-prediction side channels, and invalid curve attacks.
Provides asymmetric public-key cryptography for TARA Creator & Device identities.
"""

from typing import Tuple
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.exceptions import InvalidSignature


class Ed25519:
    """
    Hardened Ed25519 Asymmetric Digital Signature System.
    Generates 32-byte private keys (seeds), 32-byte public keys, and 64-byte signatures.
    Backed by constant-time OpenSSL implementations via PyCA cryptography.
    """
    @staticmethod
    def generate_keypair() -> Tuple[bytes, bytes]:
        """Generates a secure (private_key, public_key) pair."""
        priv_obj = ed25519.Ed25519PrivateKey.generate()
        priv_bytes = priv_obj.private_bytes_raw()
        pub_bytes = priv_obj.public_key().public_bytes_raw()
        return priv_bytes, pub_bytes

    @staticmethod
    def public_key_from_private(priv_bytes: bytes) -> bytes:
        """Derives 32-byte public key from 32-byte private key seed."""
        if len(priv_bytes) != 32:
            raise ValueError(f"Ed25519 private key seed must be exactly 32 bytes, got {len(priv_bytes)}")
        priv_obj = ed25519.Ed25519PrivateKey.from_private_bytes(priv_bytes)
        return priv_obj.public_key().public_bytes_raw()

    get_public_key = public_key_from_private

    @staticmethod
    def sign(priv_bytes: bytes, message: bytes) -> bytes:
        """Signs a message using 32-byte private key seed. Returns 64-byte signature."""
        if len(priv_bytes) != 32:
            raise ValueError(f"Ed25519 private key seed must be exactly 32 bytes, got {len(priv_bytes)}")
        priv_obj = ed25519.Ed25519PrivateKey.from_private_bytes(priv_bytes)
        return priv_obj.sign(message)

    @staticmethod
    def verify(pub_bytes: bytes, message: bytes, signature: bytes) -> bool:
        """
        Verifies a 64-byte Ed25519 signature against 32-byte public key and message.
        Returns True if cryptographically valid, False on mismatch, corruption, or wrong key.
        """
        if len(signature) != 64 or len(pub_bytes) != 32:
            return False
        try:
            pub_obj = ed25519.Ed25519PublicKey.from_public_bytes(pub_bytes)
            pub_obj.verify(signature, message)
            return True
        except (InvalidSignature, Exception):
            return False

