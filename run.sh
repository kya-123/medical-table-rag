#!/usr/bin/env bash
# RAG service launcher (Linux / macOS / WSL)
set -e
cd "$(dirname "$0")"

PY="./.venv/bin/python"
[ -x "$PY" ] || PY="python3"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export ANONYMIZED_TELEMETRY=False
export RAG_PROFILE="${RAG_PROFILE:-generic}"

[ -d "model/models" ] || echo "[warning] model/models not found - run: python3 scripts/download_models.py"

PORT="${RAG_PORT:-8001}"
echo "API docs: http://127.0.0.1:${PORT}/docs"
exec "$PY" rag_system.py
