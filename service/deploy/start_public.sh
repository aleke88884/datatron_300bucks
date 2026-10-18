#!/usr/bin/env bash
# Один запуск = сервис для жюри с публичным URL (ПЛАН §13 TODO п. 1, 4). macOS / Linux.
#   1) .venv + зависимости, 2) веса моделей (YOLOX-m, RTMPose-m), 3) service/.env с настоящим ключом,
#   4) uvicorn на :8000 в фоне, 5) Cloudflare quick tunnel → печатает URL UI/API и строки для письма жюри.
# Использование:   bash service/deploy/start_public.sh            (из корня репозитория или из service/)
#                  PORT=8000 SKIP_INSTALL=1 bash service/deploy/start_public.sh   (повторный запуск без pip)
# Остановить: Ctrl+C (останавливает и туннель, и сервер). Журнал сервера: service/data/server.log.
set -euo pipefail

SERVICE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${PORT:-8000}"
cd "$SERVICE_DIR"

# --- 1. Python и зависимости ------------------------------------------------
PY_SYS=""
for c in python3.12 python3.13 python3.11 python3; do
  if command -v "$c" >/dev/null 2>&1; then PY_SYS="$c"; break; fi
done
[ -n "$PY_SYS" ] || { echo "нужен Python 3.11+ (https://www.python.org/downloads/)"; exit 1; }
if [ ! -x .venv/bin/python ]; then
  echo "→ создаю .venv ($("$PY_SYS" --version))"
  "$PY_SYS" -m venv .venv
fi
PY="$SERVICE_DIR/.venv/bin/python"
if [ "${SKIP_INSTALL:-0}" != "1" ]; then
  echo "→ ставлю зависимости (requirements.txt)"
  "$PY" -m pip install -q --upgrade pip
  "$PY" -m pip install -q -r requirements.txt
fi

# --- 2. Веса моделей ----------------------------------------------------------
echo "→ проверяю веса моделей"
"$PY" scripts/get_models.py

# --- 3. Ключ API ----------------------------------------------------------------
if ! grep -qE '^API_KEY=.{16,}' .env 2>/dev/null || grep -qE '^API_KEY=change-me' .env 2>/dev/null; then
  echo "→ генерирую ключ API в service/.env"
  "$PY" deploy/gen_api_key.py >/dev/null
fi

# --- 4. Сервер ------------------------------------------------------------------
mkdir -p data
if curl -fsS --max-time 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "→ на :$PORT уже работает сервис — использую его (перезапустите его сами, если меняли код или ключ)"
  SERVER_PID=""
else
  echo "→ запускаю сервис на :$PORT (журнал: service/data/server.log)"
  "$PY" -m uvicorn app.main:app --host 127.0.0.1 --port "$PORT" >data/server.log 2>&1 &
  SERVER_PID=$!
fi
cleanup() { [ -n "${SERVER_PID:-}" ] && kill "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
for i in $(seq 1 120); do
  curl -fsS --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  if [ -n "${SERVER_PID:-}" ] && ! kill -0 "$SERVER_PID" 2>/dev/null; then echo "сервис упал при старте:"; tail -40 data/server.log; exit 1; fi
  sleep 1
done
curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null || { echo "сервис не поднялся за 120 с:"; tail -40 data/server.log; exit 1; }
echo "   health OK: $(curl -fsS "http://127.0.0.1:$PORT/health")"

# --- 5. cloudflared ---------------------------------------------------------------
if ! command -v cloudflared >/dev/null 2>&1; then
  if [ -x "$SERVICE_DIR/.bin/cloudflared" ]; then
    export PATH="$SERVICE_DIR/.bin:$PATH"
  elif command -v brew >/dev/null 2>&1; then
    echo "→ brew install cloudflared"; brew install cloudflared
  else
    os="$(uname -s | tr '[:upper:]' '[:lower:]')"; arch="$(uname -m)"
    case "$arch" in arm64|aarch64) arch=arm64 ;; x86_64) arch=amd64 ;; esac
    mkdir -p "$SERVICE_DIR/.bin"
    if [ "$os" = "darwin" ]; then
      url="https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-darwin-$arch.tgz"
      echo "→ скачиваю cloudflared: $url"
      curl -fL "$url" | tar -xz -C "$SERVICE_DIR/.bin"
    else
      url="https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-$arch"
      echo "→ скачиваю cloudflared: $url"
      curl -fL -o "$SERVICE_DIR/.bin/cloudflared" "$url"
    fi
    chmod +x "$SERVICE_DIR/.bin/cloudflared"
    export PATH="$SERVICE_DIR/.bin:$PATH"
  fi
fi

echo "→ поднимаю публичный туннель (адрес живёт, пока открыт этот терминал)"
echo "   ключ для письма жюри: $PY deploy/gen_api_key.py --show"
bash deploy/run_tunnel.sh "$PORT" cloudflare
