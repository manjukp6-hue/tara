#!/usr/bin/env bash
# TARA AI Core - Linux / macOS 1-Click Launcher
set -e

echo "=============================================================================="
echo "  Building and starting TARA AI Core (native Rust) & Architecture Watcher"
echo "=============================================================================="

if ! command -v cargo >/dev/null 2>&1; then
    echo "Rust toolchain (cargo) is required. Install Rust before starting TARA." >&2
    exit 1
fi

# Check and auto-start live watcher daemon if not already running
if pgrep -f "architecture_sync.*--watch" >/dev/null 2>&1; then
    echo "[Architecture Watcher] Daemon is already running in background."
else
    echo "[Architecture Watcher] Auto-starting Persistent Background Daemon..."
    if [ -f "./target/release/architecture_sync" ]; then
        ./target/release/architecture_sync --watch &
    elif [ -f "./target/debug/architecture_sync" ]; then
        ./target/debug/architecture_sync --watch &
    else
        cargo run --release --bin architecture_sync -- --watch &
    fi
    echo "[Architecture Watcher] Persistent Daemon launched."
fi

echo "Starting TARA AI Core Engine on http://127.0.0.1:${PORT:-8765} ..."
cargo run --release --bin tara_server

