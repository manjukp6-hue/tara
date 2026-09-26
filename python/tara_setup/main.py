"""
python/tara_setup/main.py

Main executable entry point for TARA Setup.
Operates 100% LOCAL ONLY with zero remote server connections.
Supports both interactive Windows desktop GUI (default) and safe local administrative CLI operations.
"""

import os
import sys
import json
import getpass
import argparse

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
python_dir = os.path.join(REPO_ROOT, "python")
if python_dir not in sys.path:
    sys.path.insert(0, python_dir)

from tara_setup.client import SetupClient


def run_cli(args: argparse.Namespace) -> int:
    client = SetupClient()

    if args.status:
        state = client.get_local_authority_state()
        print(json.dumps(state, indent=2))
        return 0 if state.get("online") else 1

    if getattr(args, "recovery_status", False) is True:
        status_info = client.get_recovery_status()
        print(f"recovery storage: {status_info['recovery_storage']}")
        print(f"recovery integrity: {status_info['recovery_integrity']}")
        print(f"trusted device recovery: {status_info['trusted_device_recovery']}")
        print(f"failed attempts: {status_info['failed_attempts']}")
        print(f"lockout status: {status_info['lockout_status']}")
        print(f"recovery history count: {status_info['recovery_history_count']}")
        return 0 if status_info["recovery_integrity"] == "VALID" else 1

    if args.list_devices:
        if not args.session_token:
            print("Error: --session-token required to list devices.", file=sys.stderr)
            return 1
        try:
            devs = client.list_devices(session_token=args.session_token)
            print(json.dumps(devs, indent=2))
            return 0
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    has_explicit_oauth_args = (
        isinstance(getattr(args, "client_id", None), str)
        or isinstance(getattr(args, "client_secret", None), str)
        or (getattr(args, "configure_oauth", False) is True)
    )
    if has_explicit_oauth_args:
        cid = getattr(args, "client_id", None) if isinstance(getattr(args, "client_id", None), str) else None
        sec = getattr(args, "client_secret", None) if isinstance(getattr(args, "client_secret", None), str) else None
        if getattr(args, "configure_oauth", False) is True and not cid:
            print("=" * 65)
            print("       CONFIGURE PROTECTED GOOGLE OAUTH CREDENTIALS")
            print("=" * 65)
            prompt_cid = input("Enter Google OAuth Client ID: ").strip()
            if not client.is_valid_client_id(prompt_cid):
                print("Error: Invalid Google OAuth Client ID format.", file=sys.stderr)
                return 1
            prompt_sec = getpass.getpass("Enter Google OAuth Client Secret (optional, press Enter to skip): ").strip() or None
            client.store_google_oauth_credentials(client_id=prompt_cid, client_secret=prompt_sec)
            print("Google OAuth credentials saved to protected local application storage.")
            return 0
        else:
            if cid and not client.is_valid_client_id(cid):
                print("Error: Invalid Google OAuth Client ID format.", file=sys.stderr)
                return 1
            client.store_google_oauth_credentials(client_id=cid, client_secret=sec)
            print("Google OAuth credentials saved to protected local application storage.")
            if not (getattr(args, "setup", False) or getattr(args, "recovery", False) or getattr(args, "register_pc2", False)):
                return 0

    if args.revoke:
        if not args.session_token:
            print("Error: --session-token required to revoke devices.", file=sys.stderr)
            return 1
        try:
            res = client.revoke_device(device_id=args.revoke, session_token=args.session_token)
            print(json.dumps(res, indent=2))
            return 0
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    if args.setup:
        # Check current local authority state
        state = client.get_local_authority_state()
        if state.get("creator_status") != "CREATOR_SETUP_REQUIRED":
            c_status = state.get("creator_status")
            print(f"Error: Local TARA Authority is already established (status: {c_status}). First-time setup cannot be re-run.", file=sys.stderr)
            return 1

        # First-time Creator Setup must use only the interactive secure flow:
        # Google OIDC/PKCE → verified identity → explicit confirmation → passphrase → local setup.
        print("=" * 65)
        print("          TARA LOCAL CREATOR SETUP — INTERACTIVE AUTHENTICATION")
        print("=" * 65)

        # 1. Real interactive Google OpenID Connect (PKCE) flow
        print("[1/4] Starting local Google OpenID Connect (PKCE) flow...")
        google_client_id = client.get_google_client_id()
        if not google_client_id or not client.is_valid_client_id(google_client_id):
            if sys.stdin.isatty():
                print("\nGoogle OAuth Client Configuration:")
                print("No Google OAuth client ID found in protected local application storage.")
                prompt_cid = input("Enter Google OAuth Client ID (or press Enter to abort): ").strip()
                if client.is_valid_client_id(prompt_cid):
                    prompt_sec = getpass.getpass("Enter Google OAuth Client Secret (optional, press Enter to skip): ").strip() or None
                    client.store_google_oauth_credentials(client_id=prompt_cid, client_secret=prompt_sec)
                    google_client_id = prompt_cid

        if not google_client_id or not client.is_valid_client_id(google_client_id):
            print(
                "Setup Error: Configured Google OAuth client ID is missing or invalid.\n"
                "Configure GOOGLE_CLIENT_ID or provide valid client credentials before starting setup.",
                file=sys.stderr
            )
            return 1

        print("Opening system browser for secure authentication...")
        try:
            flow = client.start_google_login_flow(open_browser=True)
        except Exception as e:
            print(f"Google Authentication Failed: {e}", file=sys.stderr)
            return 1
        print(f"Awaiting local loopback callback on {flow['redirect_uri']}...")
        flow["thread"].join(timeout=120)
        result_holder = flow["result_holder"]
        if not result_holder.get("token"):
            err = result_holder.get("error") or "Authentication timed out waiting for browser callback."
            print(f"Google Authentication Failed: {err}", file=sys.stderr)
            return 1
        token = result_holder["token"]
        expected_nonce = flow.get("nonce")

        # 2. Cryptographic Token & Identity Verification
        print("\n[2/4] Verifying Google identity cryptographically...")
        token_verif = client.verify_google_token_string(token, expected_nonce=expected_nonce)
        if not token_verif.get("valid"):
            print(f"Identity Verification Failed: {token_verif.get('error')}", file=sys.stderr)
            return 1

        email = token_verif["email"]
        print(f"Verified Google Account: {email}")

        # 3. Explicit Confirmation (never silent initialization)
        print("\n[3/4] Explicit Root Authority Confirmation")
        confirm_input = input(f"Confirm binding Root Creator authority to '{email}'? (yes/no): ").strip().lower()
        if confirm_input not in ("yes", "y"):
            print("Setup Aborted: Explicit Creator identity confirmation required.", file=sys.stderr)
            return 1

        # 4. Master Passphrase Confirmation
        print("\n[4/4] Master Creator Passphrase Entry")
        passphrase = getpass.getpass("Enter Master Creator Passphrase (minimum 8 characters): ")
        confirm_pass = getpass.getpass("Confirm Master Creator Passphrase: ")

        if passphrase != confirm_pass:
            print("Setup Error: Passphrases do not match.", file=sys.stderr)
            return 1

        if len(passphrase) < 8:
            print("Setup Error: Passphrase must be at least 8 characters in length.", file=sys.stderr)
            return 1

        # 5. Execute Atomic Local Creator Setup
        print("\n[*] Initializing local creator trust root and sealing authority...")
        try:
            res = client.perform_first_time_setup(
                master_passphrase=passphrase,
                confirm_passphrase=confirm_pass,
                google_id_token=token,
                confirm_identity=True,
                expected_nonce=expected_nonce
            )
            print("\n" + "=" * 65)
            print("             LOCAL CREATOR SETUP COMPLETED SUCCESSFULLY")
            print("=" * 65)
            print(f" Creator ID       : {res.get('creator_id')}")
            print(f" Display Name     : {res.get('display_name')}")
            print(f" Authority State  : {res.get('authority_state')}")
            print(f" Device Registered: {res.get('device_id')} ({res.get('device_name')})")
            print("\n Emergency Recovery Secret:")
            print(f"   >>> {res.get('recovery_code')} <<<")
            print(" Store this recovery secret safely offline.")
            print("=" * 65)
            return 0
        except Exception as e:
            print(f"Setup Error: {e}", file=sys.stderr)
            return 1

    if args.register_pc2:
        if not args.session_token:
            print("Error: --session-token required for secondary PC registration.", file=sys.stderr)
            return 1

        print("Starting local Google OpenID Connect (PKCE) flow for secondary PC registration...")
        print("Opening system browser for secure authentication...")
        flow = client.start_google_login_flow(open_browser=True)
        print(f"Awaiting local loopback callback on {flow['redirect_uri']}...")
        flow["thread"].join(timeout=120)
        result_holder = flow["result_holder"]
        if not result_holder.get("token"):
            err = result_holder.get("error") or "Authentication timed out waiting for browser callback."
            print(f"Google Authentication Failed: {err}", file=sys.stderr)
            return 1
        token = result_holder["token"]

        try:
            res = client.register_second_pc(
                google_id_token=token,
                creator_session_token=args.session_token
            )
            print(json.dumps(res, indent=2))
            return 0
        except Exception as e:
            print(f"Registration Error: {e}", file=sys.stderr)
            return 1

    if args.recovery:
        # Check current local authority state
        state = client.get_local_authority_state()
        if state.get("creator_status") == "CREATOR_SETUP_REQUIRED":
            print("Error: Local TARA Authority has not been initialized. Recovery cannot be performed on an uninitialized system.", file=sys.stderr)
            return 1

        print("=" * 65)
        print("          TARA CREATOR RECOVERY — EMERGENCY RESTORATION")
        print("=" * 65)

        # 1. Recovery Method Selection
        print("\nSelect Recovery Method:")
        print("  1) Emergency Recovery Code")
        print("  2) Google Account Recovery (OIDC/PKCE)")
        choice = input("\nEnter selection (1 or 2): ").strip()

        recovery_credential = None
        recovery_method = None
        expected_nonce = None

        if choice in ("1", "recovery_code", "code"):
            recovery_method = "recovery_code"
            recovery_credential = getpass.getpass("Enter Emergency Recovery Code: ").strip()
            if not recovery_credential:
                print("Recovery Aborted: Recovery code cannot be empty.", file=sys.stderr)
                return 1

        elif choice in ("2", "google", "google_account"):
            recovery_method = "google_account"
            print("\nStarting local Google OpenID Connect (PKCE) flow for recovery...")

            # Fail clearly before opening browser if client ID is missing or invalid
            google_client_id = client.get_google_client_id()
            if not google_client_id or not client.is_valid_client_id(google_client_id):
                if sys.stdin.isatty():
                    print("\nGoogle OAuth Client Configuration:")
                    print("No Google OAuth client ID found in protected local application storage.")
                    prompt_cid = input("Enter Google OAuth Client ID (or press Enter to abort): ").strip()
                    if client.is_valid_client_id(prompt_cid):
                        prompt_sec = getpass.getpass("Enter Google OAuth Client Secret (optional, press Enter to skip): ").strip() or None
                        client.store_google_oauth_credentials(client_id=prompt_cid, client_secret=prompt_sec)
                        google_client_id = prompt_cid

            if not google_client_id or not client.is_valid_client_id(google_client_id):
                print(
                    "Recovery Error: Configured Google OAuth client ID is missing or invalid.\n"
                    "Configure GOOGLE_CLIENT_ID or provide valid client credentials before starting Google recovery.",
                    file=sys.stderr
                )
                return 1

            print("Opening system browser for secure authentication...")
            try:
                flow = client.start_google_login_flow(open_browser=True)
            except Exception as e:
                print(f"Google Authentication Failed: {e}", file=sys.stderr)
                return 1

            print(f"Awaiting local loopback callback on {flow['redirect_uri']}...")
            flow["thread"].join(timeout=120)
            result_holder = flow["result_holder"]
            if not result_holder.get("token"):
                err = result_holder.get("error") or "Authentication timed out waiting for browser callback."
                print(f"Google Authentication Failed: {err}", file=sys.stderr)
                return 1
            recovery_credential = result_holder["token"]
            expected_nonce = flow.get("nonce")

        else:
            print("Recovery Aborted: Invalid recovery method selection.", file=sys.stderr)
            return 1

        # 2. Master Passphrase Configuration for Rotated Key
        print("\nMaster Passphrase Configuration:")
        passphrase = getpass.getpass("Enter New Master Passphrase (minimum 8 characters, or press Enter to keep default): ")
        if passphrase:
            confirm_pass = getpass.getpass("Confirm New Master Passphrase: ")
            if passphrase != confirm_pass:
                print("Recovery Error: Passphrases do not match.", file=sys.stderr)
                return 1
            if len(passphrase) < 8:
                print("Recovery Error: Passphrase must be at least 8 characters in length.", file=sys.stderr)
                return 1
        else:
            passphrase = None

        # 3. Execute Creator Key-Loss Recovery via AccessManager
        print("\n[*] Verifying authorization proof and executing recovery...")
        try:
            from TARA.ACCESS.access_manager import AccessManager
            access_mgr = AccessManager()
            recovery_kwargs = {
                "recovery_method": recovery_method,
                "recovery_credential": recovery_credential,
                "passphrase": passphrase
            }
            if expected_nonce is not None:
                recovery_kwargs["expected_nonce"] = expected_nonce
            event = access_mgr.recover_after_key_loss(**recovery_kwargs)
            print("\n" + "=" * 65)
            print("             CREATOR RECOVERY COMPLETED SUCCESSFULLY")
            print("=" * 65)
            print(f" Creator ID       : {event.get('creator_id')}")
            print(f" Display Name     : {event.get('display_name')}")
            method_desc = "Emergency Recovery Code" if (recovery_method == "recovery_code" or "recovery_code" in str(event.get('method', ''))) else "Google Account Recovery (OIDC/PKCE)"
            print(f" Recovery Method  : {method_desc}")
            print(f" Previous Version : {event.get('old_key_version')}")
            print(f" Restored Version : {event.get('new_key_version')}")
            print(f" Authority State  : ACTIVE")
            print(f" Timestamp        : {event.get('timestamp')}")
            print(" Root key rotation and recovery authorization proof verified.")
            print("=" * 65)
            return 0
        except PermissionError as pe:
            print(f"Recovery Failed: {pe}", file=sys.stderr)
            return 1
        except Exception as e:
            print(f"Recovery Error: {e}", file=sys.stderr)
            return 1

    if getattr(args, "repair_authority", False) is True:
        print("=" * 65)
        print("         TARA CREATOR AUTHORITY — ROTATED AUTHORITY REPAIR")
        print("=" * 65)

        from TARA.ACCESS.access_manager import AccessManager
        access_mgr = AccessManager()

        if access_mgr.creator.key_version < 2:
            print(f"Repair Error: Repair is only valid for already-rotated authorities (current key_version: {access_mgr.creator.key_version}).", file=sys.stderr)
            return 1

        print("\nSelect Verification Method for Authority Repair:")
        print("  1) Emergency Recovery Code")
        print("  2) Google Account Recovery (OIDC/PKCE)")
        choice = input("\nEnter selection (1 or 2): ").strip()

        recovery_credential = None
        recovery_method = None
        proof = None

        if choice in ("1", "recovery_code", "code"):
            recovery_method = "recovery_code"
            code = getpass.getpass("Enter Emergency Recovery Code: ").strip()
            if not code:
                print("Repair Aborted: Recovery code cannot be empty.", file=sys.stderr)
                return 1
            proof = access_mgr.recovery.verify_recovery_code(code)
            if not proof or not proof.is_valid():
                print("Repair Error: Invalid or expired emergency recovery code.", file=sys.stderr)
                return 1

        elif choice in ("2", "google", "google_account"):
            recovery_method = "google_account"
            print("\nStarting local Google OpenID Connect (PKCE) flow for authority repair...")
            google_client_id = client.get_google_client_id()
            if not google_client_id or not client.is_valid_client_id(google_client_id):
                if sys.stdin.isatty():
                    print("\nGoogle OAuth Client Configuration:")
                    print("No Google OAuth client ID found in protected local application storage.")
                    prompt_cid = input("Enter Google OAuth Client ID (or press Enter to abort): ").strip()
                    if client.is_valid_client_id(prompt_cid):
                        prompt_sec = getpass.getpass("Enter Google OAuth Client Secret (optional, press Enter to skip): ").strip() or None
                        client.store_google_oauth_credentials(client_id=prompt_cid, client_secret=prompt_sec)
                        google_client_id = prompt_cid

            if not google_client_id or not client.is_valid_client_id(google_client_id):
                print(
                    "Repair Error: Configured Google OAuth client ID is missing or invalid.\n"
                    "Configure GOOGLE_CLIENT_ID or provide valid client credentials before starting Google authentication.",
                    file=sys.stderr
                )
                return 1

            print("Opening system browser for secure authentication...")
            try:
                flow = client.start_google_login_flow(open_browser=True)
            except Exception as e:
                print(f"Google Authentication Failed: {e}", file=sys.stderr)
                return 1

            print(f"Awaiting local loopback callback on {flow['redirect_uri']}...")
            flow["thread"].join(timeout=120)
            result_holder = flow["result_holder"]
            if not result_holder.get("token"):
                err = result_holder.get("error") or "Authentication timed out waiting for browser callback."
                print(f"Google Authentication Failed: {err}", file=sys.stderr)
                return 1
            token = result_holder["token"]
            expected_nonce = flow.get("nonce")
            proof = access_mgr.recovery.verify_google_recovery(
                token,
                google_service=client.google_service,
                expected_nonce=expected_nonce
            )
            if not proof or not proof.is_valid():
                verif = client.google_service.verify_id_token(token, expected_nonce=expected_nonce)
                if verif.get("verified"):
                    proof = access_mgr.recovery.verify_google_recovery(verif)
            if not proof or not proof.is_valid():
                print("Repair Error: Google account verification failed for authority repair.", file=sys.stderr)
                return 1
        else:
            print("Repair Aborted: Invalid method selection.", file=sys.stderr)
            return 1

        print("\nMaster Passphrase:")
        passphrase = getpass.getpass("Enter Master Creator Passphrase (to decrypt existing rotated root key): ").strip()
        if not passphrase:
            passphrase = os.environ.get("TARA_CREATOR_PASSPHRASE")

        if not passphrase:
            print("Repair Aborted: Master Creator Passphrase is required to decrypt the rotated root key.", file=sys.stderr)
            return 1

        print("\n[*] Executing atomic authority repair with existing rotated root key...")
        try:
            res = access_mgr.repair_rotated_authority(authorization_proof=proof, passphrase=passphrase)
            print("\n" + "=" * 65)
            print("          AUTHORITY REPAIR COMPLETED SUCCESSFULLY")
            print("=" * 65)
            print(f" Creator ID       : {res.get('creator_id')}")
            print(f" Display Name     : {res.get('display_name')}")
            print(f" Key Version      : {res.get('key_version')}")
            print(f" Authority State  : {res.get('state')}")
            print(f" Timestamp        : {res.get('timestamp')}")
            print(" Authority seal synchronized and verified active.")
            print("=" * 65)
            return 0
        except Exception as e:
            print(f"Authority Repair Failed: {e}", file=sys.stderr)
            return 1

    print("No CLI action specified. Run with --help for options or run without --cli to launch graphical setup.")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="TARA Setup — Creator Trust Root & Local Device Identity"
    )
    parser.add_argument("--cli", action="store_true", help="Run in local CLI mode instead of graphical setup")
    parser.add_argument("--status", action="store_true", help="Query and print local TARA authority status")
    parser.add_argument("--recovery-status", action="store_true", help="Display safe recovery storage and integrity status")
    parser.add_argument("--setup", action="store_true", help="Execute interactive first-time creator setup")
    parser.add_argument("--recovery", action="store_true", help="Execute interactive emergency creator recovery")
    parser.add_argument("--repair-authority", action="store_true", help="Repair an already-rotated Creator authority seal using verified recovery proof without generating a new key")
    parser.add_argument("--register-pc2", action="store_true", help="Register secondary PC device with local authority")
    parser.add_argument("--session-token", help="Active creator session token for administrative operations")
    parser.add_argument("--list-devices", action="store_true", help="List registered devices from local registry")
    parser.add_argument("--revoke", help="Device ID to revoke in local registry")
    parser.add_argument("--client-id", help="Configure Google OAuth Client ID into protected local storage")
    parser.add_argument("--client-secret", help="Configure Google OAuth Client Secret into protected local storage")
    parser.add_argument("--configure-oauth", action="store_true", help="Interactively configure Google OAuth credentials into protected storage")

    args = parser.parse_args()

    # Determine execution mode:
    # If any CLI-specific flags are passed, or --cli is set
    is_cli = (
        args.cli
        or args.status
        or args.recovery_status
        or args.setup
        or args.recovery
        or getattr(args, "repair_authority", False)
        or args.register_pc2
        or args.list_devices
        or bool(args.revoke)
        or bool(args.client_id)
        or bool(args.client_secret)
        or bool(args.configure_oauth)
    )


    if is_cli:
        sys.exit(run_cli(args))
    else:
        try:
            from tara_setup.ui import launch_gui
            launch_gui()
        except Exception as e:
            print(f"Failed to launch graphical UI: {e}. Falling back to CLI mode.", file=sys.stderr)
            sys.exit(run_cli(args))


if __name__ == "__main__":
    main()
