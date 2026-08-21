#!/usr/bin/env bash
# TACHYON — Secure Deployment Script
# Deploys to Giganode cloud server (87.76.191.175) as root
# Excludes all secrets and local artifacts per security policy.

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────
REMOTE_HOST="87.76.191.175"
REMOTE_USER="root"
REMOTE_PATH="/opt/tachyon"
LOCAL_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Rsync excludes — NEVER deploy secrets, caches, or local data
RSYNC_EXCLUDES=(
    "--exclude=.env"
    "--exclude=.git"
    "--exclude=__pycache__"
    "--exclude=.pytest_cache"
    "--exclude=venv"
    "--exclude=data/ticks"
    "--exclude=logs"
    "--exclude=*.pyc"
    "--exclude=.mypy_cache"
    "--exclude=.ruff_cache"
    "--exclude=*.log"
    "--exclude=.coverage"
    "--exclude=htmlcov"
)

# ── Pre-flight checks ──────────────────────────────────────────────────────
echo "=== TACHYON Deployment ==="
echo "Local project:  ${LOCAL_PROJECT_ROOT}"
echo "Remote target:  ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}"
echo

# Verify we have the project files
if [[ ! -f "${LOCAL_PROJECT_ROOT}/requirements.txt" ]]; then
    echo "ERROR: requirements.txt not found. Run from project root."
    exit 1
fi

if [[ ! -f "${LOCAL_PROJECT_ROOT}/boot_tachyon.py" ]]; then
    echo "ERROR: boot_tachyon.py not found. Run from project root."
    exit 1
fi

# Warn if .env exists locally (should NOT be deployed)
if [[ -f "${LOCAL_PROJECT_ROOT}/.env" ]]; then
    echo "WARNING: .env exists locally and will be EXCLUDED from sync."
    echo "         Ensure production .env is already on the server."
    echo
fi

# ── Step 1: Rsync to remote ────────────────────────────────────────────────
echo "=== Step 1: Syncing project files ==="
rsync -avz --delete "${RSYNC_EXCLUDES[@]}" \
    "${LOCAL_PROJECT_ROOT}/" \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/"

echo "Sync complete."
echo

# ── Step 2: Remote setup ───────────────────────────────────────────────────
echo "=== Step 2: Remote environment setup ==="
ssh "${REMOTE_USER}@${REMOTE_HOST}" << 'REMOTE_EOF'
set -euo pipefail

cd /opt/tachyon

# Create virtual environment if missing
if [[ ! -d "venv" ]]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate and install dependencies
echo "Installing dependencies..."
source venv/bin/activate
pip install --upgrade pip wheel
pip install -r requirements.txt

# Verify critical imports
python3 -c "
import torch
import numpy
import numba
import pyarrow
import pyzmq
import msgspec
import structlog
import pydantic
import fastapi
import uvicorn
print('All critical imports OK')
"

echo "Remote setup complete."
REMOTE_EOF

echo
echo "=== Step 3: Systemd service installation ==="
# Copy systemd service file to server
scp "${LOCAL_PROJECT_ROOT}/tachyon.service" "${REMOTE_USER}@${REMOTE_HOST}:/etc/systemd/system/tachyon.service"

ssh "${REMOTE_USER}@${REMOTE_HOST}" << 'REMOTE_EOF'
set -euo pipefail

# Reload systemd and enable service
systemctl daemon-reload
systemctl enable tachyon.service

echo "Service installed and enabled."
echo "To start:    systemctl start tachyon"
echo "To status:   systemctl status tachyon"
echo "To logs:     journalctl -u tachyon -f"
REMOTE_EOF

echo
echo "=== Deployment Complete ==="
echo "Next steps on server:"
echo "  1. Ensure /opt/tachyon/.env exists with production credentials"
echo "  2. systemctl start tachyon"
echo "  3. journalctl -u tachyon -f  # follow logs"