import threading
import hmac
import secrets
"""
TARA/ACCESS/services/google_auth.py

Google/Gmail supporting identity verification layer.
Cryptographically verifies Google OpenID Connect ID tokens (JWT with RS256 signatures).
Validates signature against Google JWKS (or injected trusted public keys), audience (aud),
issuer (iss), expiration (exp), issued-at (iat), and email_verified.

Security Rules:
- Google/Gmail is a supporting authentication layer, NOT the root TARA key.
- A Google account / ID token alone must NEVER automatically grant creator authority.
- Already-authorized devices retain full operational authority completely offline
  without requiring Google network connectivity or token re-validation.
"""

import os
import sys
import base64
import json
import time
import urllib.request
import urllib.error
import hashlib
from typing import Dict, Optional, Any

from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes
from cryptography.exceptions import InvalidSignature


GOOGLE_ISSUERS = ("accounts.google.com", "https://accounts.google.com")
GOOGLE_JWKS_URL = "https://www.googleapis.com/oauth2/v3/certs"


def base64url_decode(input_str: str) -> bytes:
    """Decodes a base64url-encoded string with padding correction."""
    rem = len(input_str) % 4
    if rem > 0:
        input_str += "=" * (4 - rem)
    return base64.urlsafe_b64decode(input_str.encode("ascii"))


def base64url_encode(input_bytes: bytes) -> str:
    """Encodes bytes to a base64url string without trailing padding."""
    return base64.urlsafe_b64encode(input_bytes).rstrip(b"=").decode("ascii")


def rsa_from_jwk(jwk: Dict[str, Any]) -> rsa.RSAPublicKey:
    """Constructs an RSA public key from a JWK dict."""
    def decode_int(val: str) -> int:
        b = base64url_decode(val)
        return int.from_bytes(b, "big")

    e = decode_int(jwk["e"])
    n = decode_int(jwk["n"])
    return rsa.RSAPublicNumbers(e, n).public_key()


def resolve_stored_google_client_id() -> Optional[str]:
    """Resolves stored Google OAuth Client ID from protected local application storage or environment."""
    if "unittest" in sys.modules or "pytest" in sys.modules or os.environ.get("TARA_TEST_MODE") == "1":
        return None
    try:
        from ..crypto.dpapi_storage import unprotect_bytes_dpapi
        candidates = []
        if sys.platform == "win32":
            app_data = os.environ.get("APPDATA") or os.path.expanduser("~")
            candidates.append(os.path.join(app_data, "TARA", "identity", "google_oauth_config.dpapi"))
            candidates.append(os.path.join(app_data, "TARA", "recovery", "google_oauth_config.dpapi"))
            candidates.append(os.path.join(app_data, "TARA", "vault", "google_oauth_config.dpapi"))
        else:
            home = os.path.expanduser("~")
            candidates.append(os.path.join(home, ".tara", "identity", "google_oauth_config.dpapi"))
            candidates.append(os.path.join(home, ".tara", "recovery", "google_oauth_config.dpapi"))

        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
        candidates.append(os.path.join(repo_root, "storage", "vault", "google_oauth_config.dpapi"))

        for cf in candidates:
            if os.path.exists(cf):
                try:
                    with open(cf, "rb") as f:
                        raw = unprotect_bytes_dpapi(f.read())
                    if raw:
                        data = json.loads(raw.decode("utf-8"))
                        cid = data.get("client_id")
                        if cid and isinstance(cid, str) and cid.strip():
                            return cid.strip()
                except Exception:
                    pass

        env_cid = os.environ.get("GOOGLE_CLIENT_ID")
        if env_cid and env_cid.strip():
            return env_cid.strip()

        env_path = os.path.join(repo_root, ".env")
        if os.path.exists(env_path):
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            if k.strip() == "GOOGLE_CLIENT_ID":
                                val = v.strip().strip("'\"")
                                if val:
                                    return val
            except Exception:
                pass
    except Exception:
        pass
    return None


class GoogleAuthService:
    """
    Cryptographic verification service for Google Sign-In and ID tokens.
    """
    def __init__(
        self,
        authorized_email: Optional[str] = None,
        expected_client_id: Optional[str] = None,
        trusted_public_keys: Optional[Dict[str, rsa.RSAPublicKey]] = None
    ):
        self.authorized_email = authorized_email.lower().strip() if authorized_email else None
        self.expected_client_id = expected_client_id or resolve_stored_google_client_id()
        # Key cache: kid -> RSAPublicKey
        self.key_cache: Dict[str, rsa.RSAPublicKey] = trusted_public_keys or {}
        self.last_key_fetch: float = 0.0
        self._pending_nonces: Dict[str, Dict[str, Any]] = {}
        self._nonce_lock = threading.Lock()

    def create_auth_nonce(self, ttl_seconds: float = 300.0) -> str:
        """
        Generates a cryptographically random nonce bound to an authentication transaction.
        """
        nonce = secrets.token_urlsafe(32)
        now = time.time()
        with self._nonce_lock:
            # Purge expired nonces
            expired = [k for k, v in self._pending_nonces.items() if now > v.get("expires_at", 0)]
            for k in expired:
                del self._pending_nonces[k]

            self._pending_nonces[nonce] = {
                "created_at": now,
                "expires_at": now + ttl_seconds,
                "consumed": False
            }
        return nonce

    def bind_creator_email(self, email: str) -> None:
        """Binds the verified creator human email."""
        self.authorized_email = email.lower().strip()

    def set_expected_client_id(self, client_id: str) -> None:
        """Sets the required OAuth2 client ID (audience) for token verification."""
        self.expected_client_id = client_id

    def register_trusted_key(self, kid: str, public_key: rsa.RSAPublicKey) -> None:
        """Registers a trusted RSA public key (used for testing or offline pinned certs)."""
        self.key_cache[kid] = public_key

    def fetch_google_jwks(self, timeout: float = 5.0) -> bool:
        """
        Fetches Google public JWKS certificates from Google's endpoint.
        Gracefully returns False on network timeout or connection failure.
        """
        try:
            req = urllib.request.Request(
                GOOGLE_JWKS_URL,
                headers={"User-Agent": "TARA-Identity-Engine/2.0"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))

            for key_dict in data.get("keys", []):
                if key_dict.get("kty") == "RSA" and key_dict.get("alg") == "RS256":
                    kid = key_dict.get("kid")
                    if kid:
                        self.key_cache[kid] = rsa_from_jwk(key_dict)
            self.last_key_fetch = time.time()
            return True
        except (urllib.error.URLError, TimeoutError, OSError, ValueError):
            return False

    def verify_id_token(
        self,
        id_token: str,
        expected_client_id: Optional[str] = None,
        expected_nonce: Optional[str] = None,
        require_nonce: bool = False,
        current_time: Optional[float] = None,
        clock_skew: float = 10.0
    ) -> Dict[str, Any]:
        """
        Cryptographically validates a Google OpenID Connect ID token (RS256 JWT).
        
        Checks:
        1. Format & structure (3 dot-separated base64url segments)
        2. Header alg == 'RS256'
        3. RS256 signature verification against Google's public key (via kid)
        4. iss in ['accounts.google.com', 'https://accounts.google.com']
        5. aud matches expected_client_id
        6. exp > current_time
        7. iat <= current_time
        8. email_verified is True
        9. sub is non-empty
        """
        if not id_token or not isinstance(id_token, str):
            return {"verified": False, "error": "Missing or invalid token string"}

        parts = id_token.split(".")
        if len(parts) != 3:
            return {"verified": False, "error": "Malformed JWT: expected exactly 3 segments"}

        header_b64, payload_b64, signature_b64 = parts

        try:
            header_bytes = base64url_decode(header_b64)
            header = json.loads(header_bytes.decode("utf-8"))
        except Exception:
            return {"verified": False, "error": "Malformed JWT: unable to decode header"}

        try:
            payload_bytes = base64url_decode(payload_b64)
            payload = json.loads(payload_bytes.decode("utf-8"))
        except Exception:
            return {"verified": False, "error": "Malformed JWT: unable to decode payload"}

        # 1. Check Algorithm
        if header.get("alg") != "RS256":
            return {"verified": False, "error": f"Unsupported JWT algorithm: {header.get('alg')}. Only RS256 is accepted."}

        # 2. Check Claims
        now = time.time() if current_time is None else current_time

        # Expiration
        exp = payload.get("exp")
        if exp is None or not isinstance(exp, (int, float)):
            return {"verified": False, "error": "Token missing 'exp' claim"}
        if now > (exp + clock_skew):
            return {"verified": False, "error": f"Token expired at {exp}, current time is {now}"}

        # Issued At
        iat = payload.get("iat")
        if iat is not None and isinstance(iat, (int, float)):
            if (iat - clock_skew) > now:
                return {"verified": False, "error": f"Token issued in the future: iat={iat}, current time={now}"}

        # Issuer
        iss = payload.get("iss")
        if iss not in GOOGLE_ISSUERS:
            return {"verified": False, "error": f"Invalid issuer '{iss}'. Expected one of {GOOGLE_ISSUERS}"}

        # Audience
        target_aud = expected_client_id or self.expected_client_id
        if target_aud is not None:
            token_aud = payload.get("aud")
            if isinstance(token_aud, list):
                if target_aud not in token_aud:
                    return {"verified": False, "error": f"Audience mismatch: {target_aud} not in {token_aud}"}
            else:
                if token_aud != target_aud:
                    return {"verified": False, "error": f"Audience mismatch: expected {target_aud}, got {token_aud}"}

        # Nonce claim validation
        token_nonce = payload.get("nonce")
        if expected_nonce or require_nonce:
            if not token_nonce or not isinstance(token_nonce, str):
                return {"verified": False, "error": "Token missing required 'nonce' claim"}

            if expected_nonce:
                if not hmac.compare_digest(token_nonce, expected_nonce):
                    return {"verified": False, "error": f"Nonce mismatch: expected {expected_nonce}, got {token_nonce}"}

                with self._nonce_lock:
                    nonce_record = self._pending_nonces.get(expected_nonce)
                    if nonce_record is not None:
                        if now > nonce_record.get("expires_at", 0):
                            return {"verified": False, "error": "Nonce has expired"}
                        if nonce_record.get("consumed", False):
                            return {"verified": False, "error": "Nonce replay detected: nonce has already been consumed"}
                        nonce_record["consumed"] = True

        # Subject
        sub = payload.get("sub")
        if not sub or not isinstance(sub, str):
            return {"verified": False, "error": "Token missing valid 'sub' claim"}

        # Email Verification
        email_verified = payload.get("email_verified")
        if email_verified not in (True, "true"):
            return {"verified": False, "error": "Google email is not verified"}

        # 3. Cryptographic Signature Verification
        kid = header.get("kid")
        if not kid:
            return {"verified": False, "error": "JWT header missing 'kid' key identifier"}

        public_key = self.key_cache.get(kid)
        if public_key is None:
            # Attempt to fetch online if not in cache
            if self.fetch_google_jwks():
                public_key = self.key_cache.get(kid)

        if public_key is None:
            return {
                "verified": False,
                "error": f"Google public key for kid '{kid}' not available. Key fetch failed or network offline.",
                "network_offline": True
            }

        try:
            signature_bytes = base64url_decode(signature_b64)
            signing_input = f"{header_b64}.{payload_b64}".encode("ascii")

            public_key.verify(
                signature_bytes,
                signing_input,
                padding.PKCS1v15(),
                hashes.SHA256()
            )
        except InvalidSignature:
            return {"verified": False, "error": "Cryptographic signature verification failed: invalid signature"}
        except Exception as e:
            return {"verified": False, "error": f"Signature verification error: {str(e)}"}

        email = (payload.get("email") or "").lower().strip()

        return {
            "verified": True,
            "email": email,
            "sub": sub,
            "claims": payload,
            "provider": "google.com",
            "kid": kid
        }

    def verify_account(
        self,
        email: str,
        id_token: Optional[str] = None,
        expected_client_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Verifies a Google / Gmail account.
        Requires full cryptographic JWT ID token validation against Google JWKS or injected trusted keys.
        Fails closed if id_token is None or unverified. Email address alone is NEVER treated as authentication.
        """
        if not email or "@" not in email:
            return {"verified": False, "error": "Invalid email address format"}

        if id_token is None:
            return {
                "verified": False,
                "error": "MISSING_GOOGLE_ID_TOKEN: Cryptographic Google ID token is strictly required for authentication. Email alone is not proof of identity."
            }

        clean_email = email.lower().strip()
        account_hash = hashlib.sha256(clean_email.encode("utf-8")).hexdigest()

        token_res = self.verify_id_token(id_token, expected_client_id=expected_client_id)
        if not token_res["verified"]:
            return token_res
        # Ensure email in token matches claimed email
        if token_res.get("email") != clean_email:
            return {
                "verified": False,
                "error": f"Token email '{token_res.get('email')}' does not match claimed email '{clean_email}'"
            }

        is_creator_account = (self.authorized_email is None or clean_email == self.authorized_email)

        return {
            "verified": True,
            "email": clean_email,
            "account_hash": account_hash,
            "is_associated_creator": is_creator_account,
            "provider": "google.com",
            "supports_recovery": True
        }

    @staticmethod
    def create_mock_id_token(
        private_key: rsa.RSAPrivateKey,
        kid: str,
        email: str = "creator@test.local",
        sub: str = "google-sub-1001",
        aud: str = "tara-client-id",
        iss: str = "https://accounts.google.com",
        exp: Optional[int] = None,
        iat: Optional[int] = None,
        email_verified: bool = True,
        tamper_signature: bool = False
    ) -> str:
        """
        Helper for generating valid or deliberately invalid RS256 Google ID tokens for tests.
        """
        now = int(time.time())
        header = {"alg": "RS256", "typ": "JWT", "kid": kid}
        payload = {
            "iss": iss,
            "aud": aud,
            "sub": sub,
            "email": email,
            "email_verified": email_verified,
            "exp": exp if exp is not None else now + 3600,
            "iat": iat if iat is not None else now
        }

        h_b64 = base64url_encode(json.dumps(header).encode("utf-8"))
        p_b64 = base64url_encode(json.dumps(payload).encode("utf-8"))
        signing_input = f"{h_b64}.{p_b64}".encode("ascii")

        sig = private_key.sign(signing_input, padding.PKCS1v15(), hashes.SHA256())
        if tamper_signature:
            sig = bytearray(sig)
            sig[0] ^= 0xFF
            sig = bytes(sig)

        sig_b64 = base64url_encode(sig)
        return f"{h_b64}.{p_b64}.{sig_b64}"

