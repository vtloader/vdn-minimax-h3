#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [[ -f "$REPO_ROOT/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$REPO_ROOT/.venv/bin/activate"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
# Use ordinary resumable HTTP downloads when a Hub mirror cannot authorize Xet/CAS.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
exec "$PYTHON_BIN" -m webui.server \
    --host "${HOST:-0.0.0.0}" \
    --port "${PORT:-3000}"
