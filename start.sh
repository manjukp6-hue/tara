#!/usr/bin/env bash
# TARA AI Core - Linux / macOS 1-Click Launcher
set -e

echo "=============================================================================="
echo "  Starting TARA AI Core Production Engine"
echo "=============================================================================="

python3 -m pip install -r requirements.txt
echo "Starting TARA Web Server on http://127.0.0.1:7860 ..."
python3 app.py
