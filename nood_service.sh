#!/bin/bash
# ---------------------------------------------------------------------------
# nood_service.sh — Launch the NOOD persistent analysis service
#
# This replaces the old nood_watcher.sh pattern of spawning a new Python
# process per video.  Models are loaded once at startup and reused.
#
# Usage:
#     bash nood_service.sh
#     NOOD_PORT=8080 bash nood_service.sh
# ---------------------------------------------------------------------------
set -euo pipefail

PORT="${NOOD_PORT:-5050}"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "╔══════════════════════════════════════════════════╗"
echo "║  NOOD Analysis Service                           ║"
echo "║  Port:    $PORT                                  ║"
echo "║  Project: $PROJECT_DIR                           ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# Activate conda environment
if [ -f "$HOME/project/ls/etc/profile.d/conda.sh" ]; then
    source "$HOME/project/ls/etc/profile.d/conda.sh"
    conda activate ml
    echo "  ✓ Conda environment: ml"
elif [ -f "$PROJECT_DIR/venv/bin/activate" ]; then
    source "$PROJECT_DIR/venv/bin/activate"
    echo "  ✓ Virtualenv activated"
fi

# Ensure required directories exist
mkdir -p "$PROJECT_DIR/tmp/audio"
mkdir -p "$PROJECT_DIR/tmp/uploads"

# Run startup checks
echo "  Running startup checks..."
python "$PROJECT_DIR/startup_check.py" || true
echo ""

# Start the service
cd "$PROJECT_DIR"
exec uvicorn service:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    --workers 1 \
    --log-level info
