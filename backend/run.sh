#!/bin/bash
# Start the local development API from any directory.
set -euo pipefail
cd "$(dirname "$0")"
if [ -f .env ]; then
  set -a
  source .env
  set +a
fi
export STORAGE_ROOT="${STORAGE_ROOT:-../data}"
APP_PYTHON="${PYTHON:-.venv/bin/python}"
if [ -z "${PYTHON:-}" ] && [ ! -x "$APP_PYTHON" ]; then
  APP_PYTHON=python3
fi
mkdir -p "$STORAGE_ROOT"
echo "API: http://127.0.0.1:8000"
exec "$APP_PYTHON" -m uvicorn app.api.server:app --host 127.0.0.1 --port 8000 --reload
