#!/usr/bin/env bash
# TARA AI Core - Linux / macOS 1-Click Launcher
set -e

echo "=============================================================================="
echo "  Building and starting TARA AI Core (native Rust) Server"
echo "=============================================================================="

if ! command -v cargo >/dev/null 2>&1; then
    echo "Rust toolchain (cargo) is required. Install Rust before starting TARA." >&2
    exit 1
fi

echo "Starting TARA AI Core Engine on http://127.0.0.1:${PORT:-8765} ..."
cargo run --release --bin tara_server
