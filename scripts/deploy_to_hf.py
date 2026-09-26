#!/usr/bin/env python3
"""
scripts/deploy_to_hf.py

Production-grade automated synchronization and cryptographic verification of TARA checkpoints
to Hugging Face Hub (tara-project/tara).
Features:
- Strict pre-upload SHA256 integrity audit
- Private/public repository configuration
- Complete upload of SafeTensors, configs, tokenizers, and metadata
- Post-upload remote download and SHA256 cryptographic verification against expected hash
"""

import os
import sys
import hashlib
import argparse
import tempfile
import shutil

PROMOTED_PRODUCTION_SHA256 = "7a50308b799f2654baeafbd64dec31f088c3b07b446e198cd0f9ec2b7c0af309"

def compute_file_sha256(file_path: str) -> str:
    """Computes SHA256 checksum of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def get_hf_token(provided_token: str = None) -> str:
    """Resolves Hugging Face token from parameter, environment, or Colab userdata secrets."""
    if provided_token and provided_token.strip():
        return provided_token.strip()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if token and token.strip():
        return token.strip()

    try:
        from google.colab import userdata
        token = userdata.get("HF_TOKEN") or userdata.get("HUGGINGFACE_TOKEN")
        if token and token.strip():
            return token.strip()
    except Exception:
        pass

    return None

def push_model_to_hf(
    checkpoint_dir: str = "storage/models/tara",
    repo_id: str = "tara-project/tara",
    token: str = None,
    target_subfolder: str = None,
    private: bool = True,
    expected_sha256: str = PROMOTED_PRODUCTION_SHA256,
    verify_remote: bool = True
) -> bool:
    """Deploys model directory to Hugging Face and verifies cryptographic integrity."""
    try:
        from huggingface_hub import HfApi, create_repo, hf_hub_download
    except ImportError:
        print("[Error] huggingface_hub is not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    hf_token = get_hf_token(token)
    if not hf_token:
        print("[Error] No Hugging Face token provided. Provide via --token, set HF_TOKEN environment variable, or store in Colab Secrets.")
        sys.exit(1)

    print("=" * 70)
    print("      TARA AI HUGGING FACE HUB PRIVATE SYNCHRONIZATION & AUDIT")
    print("=" * 70 + "\n")
    print(f"Target Repository:   https://huggingface.co/{repo_id} ({'PRIVATE' if private else 'PUBLIC'})")
    print(f"Source Directory:    {checkpoint_dir}")
    print(f"Target Subfolder:    {target_subfolder if (target_subfolder and target_subfolder.strip()) else 'root'}")
    print(f"Expected SHA256:     {expected_sha256 or 'N/A'}")

    # 1. Pre-upload validation
    local_weights = os.path.join(checkpoint_dir, "model.safetensors")
    if not os.path.isfile(local_weights):
        raise FileNotFoundError(f"Missing required model weights at {local_weights}")

    local_sha = compute_file_sha256(local_weights)
    print(f"[Pre-Upload Audit] Local model.safetensors SHA256: {local_sha}")

    if expected_sha256:
        if local_sha.lower() != expected_sha256.lower():
            raise ValueError(
                f"PRE-UPLOAD AUDIT FAILED! Local model SHA256 ({local_sha}) "
                f"does not match expected promoted SHA256 ({expected_sha256})!"
            )
        print("  -> [PASS] Local weights strictly match expected promoted hash.")

    api = HfApi(token=hf_token)

    # 2. Verify or create repository
    try:
        api.repo_info(repo_id=repo_id, repo_type="model")
        print(f"[OK] Connected to Hugging Face repository '{repo_id}'.")
    except Exception:
        print(f"[Notice] Repository '{repo_id}' not found or inaccessible. Creating repository (private={private})...")
        create_repo(repo_id=repo_id, token=hf_token, repo_type="model", private=private, exist_ok=True)
        print(f"[OK] Repository '{repo_id}' verified/created.")

    # 3. Upload model artifacts
    print("\nUploading model artifacts (SafeTensors, configs, manifests)...")
    path_in_repo = target_subfolder.strip("/") if (target_subfolder and target_subfolder.strip()) else None
    api.upload_folder(
        folder_path=checkpoint_dir,
        repo_id=repo_id,
        path_in_repo=path_in_repo,
        commit_message=f"deploy: upload {os.path.basename(checkpoint_dir)} verified neural weights [SHA: {local_sha[:12]}]"
    )

    dest_url = f"https://huggingface.co/{repo_id}"
    if path_in_repo:
        dest_url += f"/tree/main/{path_in_repo}"
    print(f"[OK] Files uploaded to {dest_url}")

    # 4. Post-upload cryptographic verification
    if verify_remote:
        print("\n" + "-" * 70)
        print("POST-UPLOAD CRYPTOGRAPHIC INTEGRITY VERIFICATION")
        print("-" * 70)
        remote_filename = "model.safetensors"
        if path_in_repo:
            remote_filename = f"{path_in_repo}/model.safetensors"

        print(f"Downloading remote '{remote_filename}' from '{repo_id}' to verify hash...")
        temp_dir = tempfile.mkdtemp(prefix="hf_verify_")
        try:
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=remote_filename,
                token=hf_token,
                repo_type="model",
                force_download=True,
                local_dir=temp_dir
            )
            remote_sha = compute_file_sha256(downloaded_path)
            print(f"[Post-Upload Audit] Remote model.safetensors SHA256: {remote_sha}")

            target_check_sha = expected_sha256 or local_sha
            if remote_sha.lower() != target_check_sha.lower():
                raise ValueError(
                    f"POST-UPLOAD VERIFICATION FAILED! Remote SHA256 ({remote_sha}) "
                    f"does not match expected ({target_check_sha})!"
                )
            print(f"  -> [PASS] Remote model weights SHA256 matches expected ({target_check_sha})!")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    print("\n" + "=" * 70)
    print("[SUCCESS] HUGGING FACE PRIVATE SYNCHRONIZATION AND SHA256 AUDIT COMPLETE!")
    print(f"Model URL:           {dest_url}")
    print(f"Verified SHA256:     {local_sha}")
    print("=" * 70 + "\n")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push TARA model checkpoint to Hugging Face with cryptographic audit")
    parser.add_argument("--checkpoint-dir", default="storage/models/tara", help="Local checkpoint directory")
    parser.add_argument("--repo-id", default="tara-project/tara", help="Target Hugging Face repository ID")
    parser.add_argument("--token", default=None, help="Hugging Face write token")
    parser.add_argument("--subfolder", default="", help="Target subfolder in HF repo (empty for root)")
    parser.add_argument("--public", action="store_true", help="Make repository public (default: private)")
    parser.add_argument("--expected-sha256", default=PROMOTED_PRODUCTION_SHA256, help="Expected SHA256 of model.safetensors")
    parser.add_argument("--no-verify", action="store_true", help="Skip remote download verification")
    args = parser.parse_args()

    push_model_to_hf(
        checkpoint_dir=args.checkpoint_dir,
        repo_id=args.repo_id,
        token=args.token,
        target_subfolder=args.subfolder if args.subfolder else None,
        private=not args.public,
        expected_sha256=args.expected_sha256,
        verify_remote=not args.no_verify
    )
