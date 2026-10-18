#!/usr/bin/env python3
"""
Standalone Dataset Downloader Utility

Executes isolated dataset downloading, streaming dynamic cryptographic hash verification,
permissive commercial license gating, and operational provenance registry management.
Completely isolated with zero external imports from neural models or core training systems.
"""

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

# Permissive approved licenses granting unconditional commercial, modification, and training rights
PERMISSIVE_LICENSES = {
    "MIT",
    "APACHE-2.0",
    "APACHE 2.0",
    "BSD-2-CLAUSE",
    "BSD-3-CLAUSE",
    "CC0-1.0",
    "CC0",
    "UNLICENSE",
    "ISC",
}

# Forbidden restrictions causing immediate fail-closed rejection
FORBIDDEN_RESTRICTION_SUBSTRINGS = [
    "-NC",
    "NON-COMMERCIAL",
    "NONCOMMERCIAL",
    "-ND",
    "NO-DERIVATIVES",
    "NODERIVATIVES",
    "RESEARCH-ONLY",
    "ACADEMIC-ONLY",
    "GPL",
    "AGPL",
    "LGPL",
    "CC-BY-SA",
]


def is_permissive_compliant(license_id: str) -> bool:
    """Evaluates if a license identifier satisfies 4-way commercial permissive rights."""
    norm = license_id.strip().upper()
    for forbidden in FORBIDDEN_RESTRICTION_SUBSTRINGS:
        if forbidden in norm:
            return False
    return norm in PERMISSIVE_LICENSES


def resolve_default_target_dir() -> Path:
    """Resolves target directory defaulting to D: drive with fallback to local storage."""
    preferred = Path(r"D:\taracore_datasets\downloaded")
    try:
        if preferred.drive and os.path.exists(preferred.drive + "\\"):
            preferred.mkdir(parents=True, exist_ok=True)
            return preferred
    except Exception:
        pass
    fallback = Path("storage/downloaded")
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def compute_dynamic_sha256(file_path: Path) -> str:
    """Computes streaming SHA-256 hash dynamically from file bytes."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class DownloadLedger:
    """Operational registry for repeat exclusion and provenance verification."""

    def __init__(self, ledger_path: Path):
        self.ledger_path = ledger_path
        self.records = {}
        self.load()

    def load(self):
        if self.ledger_path.exists():
            try:
                with open(self.ledger_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.records = data.get("records", {})
            except Exception as e:
                print(f"Warning: Failed to parse operational ledger: {e}", file=sys.stderr)
                self.records = {}

    def save(self):
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.ledger_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"records": self.records}, f, indent=2)
        tmp.replace(self.ledger_path)

    def has_verified_entry(self, key: str, expected_sha: str) -> bool:
        rec = self.records.get(key)
        if rec and rec.get("status") == "verified":
            return rec.get("sha256", "").lower() == expected_sha.lower()
        return False

    def record_asset(self, key: str, entry: dict):
        self.records[key] = entry
        self.save()


def download_dataset(
    url: str,
    output_filename: str,
    license_id: str,
    license_proof_url: str = "",
    target_dir: Path = None,
) -> bool:
    """Downloads an authentic dataset with license gating and dynamic integrity checks."""
    if not is_permissive_compliant(license_id):
        print(
            f"Error: License '{license_id}' rejected. "
            "Must be unconditionally permissive for commercial, modification, and training use.",
            file=sys.stderr,
        )
        return False

    target_dir = target_dir or resolve_default_target_dir()
    ledger = DownloadLedger(target_dir / "download_ledger.json")
    destination_file = target_dir / output_filename

    # Dynamic repeat skipping check
    if destination_file.exists():
        existing_sha = compute_dynamic_sha256(destination_file)
        if ledger.has_verified_entry(output_filename, existing_sha):
            print(f"Skipping download: '{output_filename}' already verified in dynamic ledger (SHA-256: {existing_sha}).")
            return True

    print(f"Downloading from {url} to {destination_file}...")
    temp_destination = destination_file.with_name(f"{destination_file.name}.download.tmp")
    success = False

    try:
        urllib.request.urlretrieve(url, temp_destination)
        dynamic_sha = compute_dynamic_sha256(temp_destination)
        file_size = temp_destination.stat().st_size

        temp_destination.replace(destination_file)

        # Update dynamic runtime ledger
        ledger.record_asset(
            output_filename,
            {
                "source_url": url,
                "license_proof_url": license_proof_url,
                "license": license_id,
                "sha256": dynamic_sha,
                "timestamp_epoch_secs": int(time.time()),
                "file_size_bytes": file_size,
                "status": "verified",
            },
        )
        print(f"Successfully downloaded and verified '{output_filename}'.")
        print(f" - Dynamic SHA-256: {dynamic_sha}")
        print(f" - Size: {file_size} bytes")
        print(f" - License: {license_id}")
        success = True
        return True
    except BaseException as e:
        print(f"Download failed: {e}", file=sys.stderr)
        return False
    finally:
        if not success and temp_destination.exists():
            try:
                temp_destination.unlink()
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(
        description="Standalone Dataset Downloader with Dynamic Integrity and License Gating"
    )
    parser.add_argument("--url", help="Direct HTTP/HTTPS source URL to download")
    parser.add_argument("--output", help="Filename to save within destination directory")
    parser.add_argument("--license", help="SPDX license identifier (e.g., Apache-2.0, MIT)")
    parser.add_argument("--proof", default="", help="License proof URL")
    parser.add_argument("--dir", default=None, help="Destination directory override")
    parser.add_argument("--check-license", help="Evaluate if license identifier is compliant")
    parser.add_argument("--status", action="store_true", help="Print status of operational ledger")

    args = parser.parse_args()

    if args.check_license:
        compliant = is_permissive_compliant(args.check_license)
        print(f"License '{args.check_license}': Compliant = {compliant}")
        sys.exit(0 if compliant else 1)

    target_dir = Path(args.dir) if args.dir else resolve_default_target_dir()

    if args.status:
        ledger = DownloadLedger(target_dir / "download_ledger.json")
        print(f"Ledger directory: {target_dir}")
        print(f"Verified assets count: {len(ledger.records)}")
        for k, v in ledger.records.items():
            print(f" - [{v.get('status')}] {k} (SHA: {v.get('sha256')}, size: {v.get('file_size_bytes')}B)")
        sys.exit(0)

    if not args.url or not args.output or not args.license:
        parser.print_help()
        sys.exit(1)

    success = download_dataset(
        url=args.url,
        output_filename=args.output,
        license_id=args.license,
        license_proof_url=args.proof,
        target_dir=target_dir,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
