#!/usr/bin/env bash
set -euo pipefail

cd /opt/xagent

# Start FastAPI server directly
# Database initialization and migrations will be handled by init_db() in app.py
BACKEND_PORT="${BACKEND_PORT:-8000}"
BACKEND_HOST="${BACKEND_HOST:-0.0.0.0}"
echo "Starting Xagent web server on ${BACKEND_HOST}:${BACKEND_PORT}..."
exec xagent-web --host "$BACKEND_HOST" --port "$BACKEND_PORT"
