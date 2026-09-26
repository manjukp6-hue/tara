"""
tests/test_cli_recovery.py

Comprehensive production validation for interactive --recovery entry point in python/tara_setup/main.py.
Validates:
1. Rejection on uninitialized authority state
2. Interactive method selection (Invalid selection -> abort)
3. Emergency recovery code path with valid code -> RecoveryAuthorizationProof -> key-loss recovery
4. Emergency recovery code path with invalid code -> rejection / PermissionError
5. Google account recovery path with verified token -> RecoveryAuthorizationProof -> key-loss recovery
6. Zero leakage of recovery secrets in output or logs
"""

import os
import sys
import io
import json
import unittest
from unittest.mock import patch, MagicMock

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
python_dir = os.path.join(REPO_ROOT, "python")
if python_dir not in sys.path:
    sys.path.insert(0, python_dir)

from tara_setup import main as cli_main
from TARA.ACCESS.restore.restore_manager import RecoveryAuthorizationProof


class TestCliRecoveryEntryPoint(unittest.TestCase):

    @patch("tara_setup.main.SetupClient")
    def test_recovery_aborts_if_uninitialized(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.get_local_authority_state.return_value = {"creator_status": "CREATOR_SETUP_REQUIRED"}
        mock_client_cls.return_value = mock_client

        args = MagicMock()
        args.recovery = True
        args.recovery_status = False
        args.status = False
        args.setup = False
        args.register_pc2 = False
        args.list_devices = False
        args.revoke = None
        args.session_token = None

        stderr_buf = io.StringIO()
        with patch("sys.stderr", stderr_buf):
            ret = cli_main.run_cli(args)

        self.assertEqual(ret, 1)
        self.assertIn("Local TARA Authority has not been initialized", stderr_buf.getvalue())

    @patch("tara_setup.main.SetupClient")
    def test_recovery_aborts_on_invalid_choice(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.get_local_authority_state.return_value = {"creator_status": "ACTIVE"}
        mock_client_cls.return_value = mock_client

        args = MagicMock()
        args.recovery = True
        args.recovery_status = False
        args.status = False
        args.setup = False
        args.register_pc2 = False
        args.list_devices = False
        args.revoke = None
        args.session_token = None

        with patch("builtins.input", return_value="invalid_choice"), \
             patch("sys.stderr", io.StringIO()) as err_buf:
            ret = cli_main.run_cli(args)

        self.assertEqual(ret, 1)
        self.assertIn("Invalid recovery method selection", err_buf.getvalue())

    @patch("tara_setup.main.SetupClient")
    @patch("TARA.ACCESS.access_manager.AccessManager")
    def test_recovery_code_success(self, mock_access_mgr_cls, mock_client_cls):
        mock_client = MagicMock()
        mock_client.get_local_authority_state.return_value = {"creator_status": "ACTIVE"}
        mock_client_cls.return_value = mock_client

        mock_access = MagicMock()
        mock_access.recover_after_key_loss.return_value = {
            "creator_id": "ROOT_OPERATOR",
            "display_name": "OPERATOR_ROOT",
            "old_key_version": 1,
            "new_key_version": 2,
            "timestamp": "2026-09-25T12:00:00Z"
        }
        mock_access_mgr_cls.return_value = mock_access

        args = MagicMock()
        args.recovery = True
        args.recovery_status = False
        args.status = False
        args.setup = False
        args.register_pc2 = False
        args.list_devices = False
        args.revoke = None
        args.session_token = None

        # input choice: '1'
        # getpass: 1) recovery code, 2) passphrase, 3) confirm passphrase
        secret_code = "ABCD-1234-EFGH-5678"
        with patch("builtins.input", return_value="1"), \
             patch("getpass.getpass", side_effect=[secret_code, "secretpassphrase123", "secretpassphrase123"]), \
             patch("sys.stdout", io.StringIO()) as out_buf:
            ret = cli_main.run_cli(args)

        self.assertEqual(ret, 0)
        output = out_buf.getvalue()
        self.assertIn("CREATOR RECOVERY COMPLETED SUCCESSFULLY", output)
        self.assertIn("Restored Version : 2", output)
        # Verify secret recovery code is NEVER leaked in output
        self.assertNotIn(secret_code, output)
        self.assertNotIn("secretpassphrase123", output)

        mock_access.recover_after_key_loss.assert_called_once_with(
            recovery_method="recovery_code",
            recovery_credential=secret_code,
            passphrase="secretpassphrase123"
        )

    @patch("tara_setup.main.SetupClient")
    @patch("TARA.ACCESS.access_manager.AccessManager")
    def test_recovery_code_failure(self, mock_access_mgr_cls, mock_client_cls):
        mock_client = MagicMock()
        mock_client.get_local_authority_state.return_value = {"creator_status": "ACTIVE"}
        mock_client_cls.return_value = mock_client

        mock_access = MagicMock()
        mock_access.recover_after_key_loss.side_effect = PermissionError("Recovery verification failed: invalid credentials or authorization proof.")
        mock_access_mgr_cls.return_value = mock_access

        args = MagicMock()
        args.recovery = True
        args.recovery_status = False
        args.status = False
        args.setup = False
        args.register_pc2 = False
        args.list_devices = False
        args.revoke = None
        args.session_token = None

        wrong_code = "WRONG-CODE-0000"
        with patch("builtins.input", return_value="1"), \
             patch("getpass.getpass", side_effect=[wrong_code, "", ""]), \
             patch("sys.stderr", io.StringIO()) as err_buf:
            ret = cli_main.run_cli(args)

        self.assertEqual(ret, 1)
        err = err_buf.getvalue()
        self.assertIn("Recovery Failed: Recovery verification failed", err)
        self.assertNotIn(wrong_code, err)

    @patch("tara_setup.main.SetupClient")
    @patch("TARA.ACCESS.access_manager.AccessManager")
    def test_google_recovery_success(self, mock_access_mgr_cls, mock_client_cls):
        mock_thread = MagicMock()
        result_holder = {"token": "valid.google.id_token"}

        mock_client = MagicMock()
        mock_client.get_local_authority_state.return_value = {"creator_status": "ACTIVE"}
        mock_client.get_google_client_id.return_value = "123456789-test.apps.googleusercontent.com"
        mock_client.is_valid_client_id.return_value = True
        mock_client.start_google_login_flow.return_value = {
            "redirect_uri": "http://127.0.0.1:8085/callback",
            "thread": mock_thread,
            "result_holder": result_holder
        }
        mock_client_cls.return_value = mock_client

        mock_access = MagicMock()
        mock_access.recover_after_key_loss.return_value = {
            "creator_id": "ROOT_OPERATOR",
            "display_name": "OPERATOR_ROOT",
            "old_key_version": 2,
            "new_key_version": 3,
            "timestamp": "2026-09-25T12:05:00Z"
        }
        mock_access_mgr_cls.return_value = mock_access

        args = MagicMock()
        args.recovery = True
        args.recovery_status = False
        args.status = False
        args.setup = False
        args.register_pc2 = False
        args.list_devices = False
        args.revoke = None
        args.session_token = None

        with patch("builtins.input", return_value="2"), \
             patch("getpass.getpass", return_value=""), \
             patch("sys.stdout", io.StringIO()) as out_buf:
            ret = cli_main.run_cli(args)

        self.assertEqual(ret, 0)
        output = out_buf.getvalue()
        self.assertIn("CREATOR RECOVERY COMPLETED SUCCESSFULLY", output)
        self.assertIn("Restored Version : 3", output)
        self.assertNotIn("valid.google.id_token", output)

        mock_access.recover_after_key_loss.assert_called_once_with(
            recovery_method="google_account",
            recovery_credential="valid.google.id_token",
            passphrase=None
        )

    @patch("tara_setup.main.SetupClient")
    def test_google_recovery_missing_client_id_aborts_cleanly(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.get_local_authority_state.return_value = {"creator_status": "ACTIVE"}
        mock_client.get_google_client_id.return_value = None
        mock_client.is_valid_client_id.return_value = False
        mock_client_cls.return_value = mock_client

        args = MagicMock()
        args.recovery = True
        args.recovery_status = False
        args.status = False
        args.setup = False
        args.register_pc2 = False
        args.list_devices = False
        args.revoke = None
        args.session_token = None

        with patch("builtins.input", return_value="2"), \
             patch("sys.stderr", io.StringIO()) as err_buf:
            ret = cli_main.run_cli(args)

        self.assertEqual(ret, 1)
        err = err_buf.getvalue()
        self.assertIn("Configured Google OAuth client ID is missing or invalid", err)
        mock_client.start_google_login_flow.assert_not_called()

    @patch("tara_setup.main.SetupClient")
    def test_recovery_status_cli_valid(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.get_recovery_status.return_value = {
            "recovery_storage": "PROTECTED",
            "recovery_integrity": "VALID",
            "trusted_device_recovery": "ENABLED",
            "failed_attempts": 0,
            "lockout_status": "UNLOCKED",
            "recovery_history_count": 2,
        }
        mock_client_cls.return_value = mock_client

        args = MagicMock()
        args.recovery = False
        args.recovery_status = True
        args.status = False
        args.setup = False
        args.register_pc2 = False
        args.list_devices = False
        args.revoke = None
        args.session_token = None

        with patch("sys.stdout", io.StringIO()) as out_buf:
            ret = cli_main.run_cli(args)

        self.assertEqual(ret, 0)
        output = out_buf.getvalue()
        self.assertIn("recovery storage: PROTECTED", output)
        self.assertIn("recovery integrity: VALID", output)
        self.assertIn("trusted device recovery: ENABLED", output)
        self.assertIn("failed attempts: 0", output)
        self.assertIn("lockout status: UNLOCKED", output)
        self.assertIn("recovery history count: 2", output)

    @patch("tara_setup.main.SetupClient")
    def test_recovery_status_cli_invalid(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.get_recovery_status.return_value = {
            "recovery_storage": "PROTECTED",
            "recovery_integrity": "INVALID",
            "trusted_device_recovery": "DISABLED",
            "failed_attempts": 0,
            "lockout_status": "LOCKED",
            "recovery_history_count": 0,
        }
        mock_client_cls.return_value = mock_client

        args = MagicMock()
        args.recovery = False
        args.recovery_status = True
        args.status = False
        args.setup = False
        args.register_pc2 = False
        args.list_devices = False
        args.revoke = None
        args.session_token = None

        with patch("sys.stdout", io.StringIO()) as out_buf:
            ret = cli_main.run_cli(args)

        self.assertEqual(ret, 1)
        output = out_buf.getvalue()
        self.assertIn("recovery storage: PROTECTED", output)
        self.assertIn("recovery integrity: INVALID", output)
        self.assertIn("trusted device recovery: DISABLED", output)
        self.assertIn("failed attempts: 0", output)
        self.assertIn("lockout status: LOCKED", output)
        self.assertIn("recovery history count: 0", output)

    def test_recovery_status_live_loader(self):
        from tara_setup.client import SetupClient
        client = SetupClient()
        status = client.get_recovery_status()
        self.assertIn(status["recovery_storage"], ("PROTECTED", "UNPROTECTED"))
        self.assertIn(status["recovery_integrity"], ("VALID", "INVALID"))
        self.assertIn(status["trusted_device_recovery"], ("ENABLED", "DISABLED"))
        self.assertIsInstance(status["failed_attempts"], int)
        self.assertIn(status["lockout_status"], ("UNLOCKED", "LOCKED"))
        self.assertIsInstance(status["recovery_history_count"], int)
        # Ensure zero secret leakage in dictionary keys
        forbidden_keys = {"recovery_code", "recovery_code_hash", "recovery_salt", "recovery_email", "root_public_key", "passphrase", "token", "ciphertext"}
        self.assertEqual(set(status.keys()) & forbidden_keys, set())

    def test_oauth_credentials_stored_and_retrieved_from_protected_storage(self):
        import tempfile
        import shutil
        from tara_setup.client import SetupClient

        temp_dir = tempfile.mkdtemp()
        try:
            client = SetupClient(storage_dir=temp_dir)
            test_cid = "123456789012-testclient.apps.googleusercontent.com"
            test_sec = "GOCSPX-SecretForTesting123"

            # Store credentials
            client.store_google_oauth_credentials(client_id=test_cid, client_secret=test_sec)

            # Files must exist on disk
            cid_file = os.path.join(temp_dir, "google_oauth_client_id.dpapi")
            sec_file = os.path.join(temp_dir, "google_oauth_secret.dpapi")
            cfg_file = os.path.join(temp_dir, "google_oauth_config.dpapi")
            self.assertTrue(os.path.exists(cid_file))
            self.assertTrue(os.path.exists(sec_file))
            self.assertTrue(os.path.exists(cfg_file))

            # Retrieve via a fresh client instance pointing to same storage
            client2 = SetupClient(storage_dir=temp_dir)
            # Ensure neither env nor repo files are used
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(client2.get_google_client_id(), test_cid)
                self.assertEqual(client2.get_google_client_secret(), test_sec)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_oauth_credentials_auto_persisted_from_env_to_protected_storage(self):
        import tempfile
        import shutil
        from tara_setup.client import SetupClient

        temp_dir = tempfile.mkdtemp()
        try:
            test_cid = "987654321098-envclient.apps.googleusercontent.com"
            test_sec = "GOCSPX-EnvSecret12345"

            # Initialize client with empty storage
            client = SetupClient(storage_dir=temp_dir)
            self.assertIsNone(client.get_google_client_id())

            # Now provide via environment variables
            with patch.dict(os.environ, {"GOOGLE_CLIENT_ID": test_cid, "GOOGLE_CLIENT_SECRET": test_sec}):
                loaded_cid = client.get_google_client_id()
                loaded_sec = client.get_google_client_secret()
                self.assertEqual(loaded_cid, test_cid)
                self.assertEqual(loaded_sec, test_sec)

            # Clear environment completely — credentials must now be in protected local storage
            with patch.dict(os.environ, {}, clear=True):
                client2 = SetupClient(storage_dir=temp_dir)
                self.assertEqual(client2.get_google_client_id(), test_cid)
                self.assertEqual(client2.get_google_client_secret(), test_sec)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_recovery_automatically_uses_stored_oauth_credentials_without_env_or_reentry(self):
        import tempfile
        import shutil
        from tara_setup.client import SetupClient

        temp_dir = tempfile.mkdtemp()
        try:
            stored_cid = "555555555555-stored.apps.googleusercontent.com"
            stored_sec = "GOCSPX-StoredSecret555"

            # Pre-populate protected storage as if Creator Setup had run
            client = SetupClient(storage_dir=temp_dir)
            client.store_google_oauth_credentials(client_id=stored_cid, client_secret=stored_sec)

            # Test that run_cli --recovery uses the stored client_id without environment variables
            mock_thread = MagicMock()
            result_holder = {"token": "recovered.google.id_token"}

            with patch.dict(os.environ, {}, clear=True), \
                 patch("tara_setup.main.SetupClient") as mock_client_cls, \
                 patch("TARA.ACCESS.access_manager.AccessManager") as mock_access_mgr_cls:

                mock_client = SetupClient(storage_dir=temp_dir)
                # Mock start_google_login_flow so browser doesn't open
                mock_client.start_google_login_flow = MagicMock(return_value={
                    "redirect_uri": "http://127.0.0.1:8085/callback",
                    "thread": mock_thread,
                    "result_holder": result_holder
                })
                mock_client_cls.return_value = mock_client

                mock_access = MagicMock()
                mock_access.recover_after_key_loss.return_value = {
                    "creator_id": "ROOT_OPERATOR",
                    "display_name": "OPERATOR_ROOT",
                    "old_key_version": 1,
                    "new_key_version": 2,
                    "timestamp": "2026-09-25T13:00:00Z"
                }
                mock_access_mgr_cls.return_value = mock_access

                args = MagicMock()
                args.recovery = True
                args.recovery_status = False
                args.status = False
                args.setup = False
                args.register_pc2 = False
                args.list_devices = False
                args.revoke = None
                args.session_token = None
                args.client_id = None
                args.client_secret = None
                args.configure_oauth = False

                with patch("builtins.input", return_value="2"), \
                     patch("getpass.getpass", return_value=""), \
                     patch("sys.stdout", io.StringIO()) as out_buf:
                    ret = cli_main.run_cli(args)

                self.assertEqual(ret, 0)
                # start_google_login_flow was called because stored_cid was loaded automatically!
                mock_client.start_google_login_flow.assert_called_once_with(open_browser=True)
                mock_access.recover_after_key_loss.assert_called_once_with(
                    recovery_method="google_account",
                    recovery_credential="recovered.google.id_token",
                    passphrase=None
                )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

