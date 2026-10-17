#!/usr/bin/env bash
# Полный прогон e2e-тестов. Использование:
#   scripts/e2e.sh              — всё, включая Docker (если демон доступен)
#   scripts/e2e.sh --no-docker  — без Docker-тестов
#   scripts/e2e.sh -k lifecycle — любые аргументы pytest пробрасываются
set -euo pipefail
SERVICE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SERVICE_DIR"
PY="$SERVICE_DIR/.venv/bin/python"; [ -x "$PY" ] || PY=python3
ARGS=()
for a in "$@"; do
  if [ "$a" = "--no-docker" ]; then export E2E_SKIP_DOCKER=1; else ARGS+=("$a"); fi
done
"$PY" -m pip install -q -r requirements-dev.txt
# "${ARGS[@]+...}" — пустой массив при set -u в bash 3.2 (macOS) иначе даёт «unbound variable»
exec "$PY" -m pytest ${ARGS[@]+"${ARGS[@]}"}
