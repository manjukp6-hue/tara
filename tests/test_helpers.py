import os
import time
from typing import Optional
from cryptography.hazmat.primitives.asymmetric import rsa
from TARA.ACCESS.services.google_auth import GoogleAuthService

_TEST_RSA_KEY: Optional[rsa.RSAPrivateKey] = None
_TEST_KID = 'tara-test-env-kid'

def get_test_rsa_key() -> rsa.RSAPrivateKey:
    global _TEST_RSA_KEY
    if _TEST_RSA_KEY is None:
        _TEST_RSA_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return _TEST_RSA_KEY

def create_test_id_token(email: str = 'creator@test.local', kid: str = _TEST_KID) -> str:
    return GoogleAuthService.create_mock_id_token(
        private_key=get_test_rsa_key(),
        kid=kid,
        email=email
    )

def setup_test_identity_manager(id_mgr):
    os.environ['TARA_TEST_MODE'] = '1'
    os.environ['TARA_CREATOR_PASSPHRASE'] = 'tara_test_suite_passphrase_env_12345'
    os.environ['TARA_DEVICE_SECRET'] = 'tara_test_suite_passphrase_env_12345'

    id_mgr.biometric_service.test_mode = True
    id_mgr.google_service.register_trusted_key(_TEST_KID, get_test_rsa_key().public_key())

    orig_first_setup = id_mgr.first_creator_setup
    def wrapped_first_setup(*args, **kwargs):
        if 'google_id_token' not in kwargs or kwargs['google_id_token'] is None:
            email = kwargs.get('google_email', args[0] if len(args) > 0 else 'creator@test.local')
            kwargs['google_id_token'] = create_test_id_token(email)
        return orig_first_setup(*args, **kwargs)
    id_mgr.first_creator_setup = wrapped_first_setup

    orig_reg_device = id_mgr.register_new_device
    def wrapped_reg_device(*args, **kwargs):
        if 'google_id_token' not in kwargs or kwargs['google_id_token'] is None:
            email = kwargs.get('google_email', args[0] if len(args) > 0 else 'creator@test.local')
            kwargs['google_id_token'] = create_test_id_token(email)
        return orig_reg_device(*args, **kwargs)
    id_mgr.register_new_device = wrapped_reg_device

    return id_mgr