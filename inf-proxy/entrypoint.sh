#!/usr/bin/env bash
set -euo pipefail

# Derive the nginx upstream for non-chat routes from the model base URL
# (strip a trailing /v1) unless MODEL_UPSTREAM is set explicitly.
if [ -z "${MODEL_UPSTREAM:-}" ]; then
  MODEL_UPSTREAM="${INF_PROXY_MODEL_BASE_URL:-http://dummy-model:8000/v1}"
  MODEL_UPSTREAM="${MODEL_UPSTREAM%/v1}"
  MODEL_UPSTREAM="${MODEL_UPSTREAM%/}"
fi
export MODEL_UPSTREAM
echo "[entrypoint] nginx model upstream: ${MODEL_UPSTREAM}"

envsubst '${MODEL_UPSTREAM}' < /app/nginx.conf.template > /etc/nginx/conf.d/default.conf

# FastAPI app on localhost; nginx fronts it on :8000.
uvicorn app.main:app --host 127.0.0.1 --port "${INF_PROXY_APP_PORT:-9000}" &
UVICORN_PID=$!

trap 'kill -TERM "$UVICORN_PID" 2>/dev/null || true' TERM INT

exec nginx -g 'daemon off;'
