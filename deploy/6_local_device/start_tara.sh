#!/usr/bin/env bash
# TARA AI Core - Linux / macOS Local 1-Click Launcher
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"
cd "$DIR"

echo "=============================================================================="
echo "  TARA AI Core - Starting Local Production Engine"
echo "=============================================================================="

python3 -m pip install -r requirements.txt
echo "Starting TARA Web Server on http://127.0.0.1:7860 ..."
python3 app.py
