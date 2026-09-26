#!/usr/bin/env bash
# ==============================================================================
# TARA AI Core - Automated Production VPS Installer (Ubuntu/Debian)
# ==============================================================================
set -e

echo "=== Installing TARA AI Core on Linux VPS ==="

# 1. Update system packages
apt-get update -y
apt-get install -y python3 python3-pip python3-venv git curl nginx

# 2. Setup application directory
INSTALL_DIR="/opt/tara"
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

# 3. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 4. Install dependencies
pip install --upgrade pip
pip install fastapi uvicorn safetensors cryptography pydantic requests

# 5. Setup systemd service
cp deploy/5_vps_iaas/tara.service /etc/systemd/system/tara.service
systemctl daemon-reload
systemctl enable tara
systemctl restart tara

echo "=== TARA Service is Active and Running on Port 7860! ==="
echo "Status check: systemctl status tara"
