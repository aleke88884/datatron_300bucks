#!/usr/bin/env bash
# Публичный URL для локального сервиса: Cloudflare quick tunnel (по умолчанию) или ngrok.
# Использование:
#   deploy/run_tunnel.sh [PORT] [cloudflare|ngrok]
#   PORT по умолчанию 8000 (uvicorn / docker compose); smoke.sh использует 8010.
# Печатает публичный URL и строки для письма жюри; туннель работает, пока не нажать Ctrl+C.
# Обработка видео остаётся локальной: туннель только проксирует HTTP к localhost:PORT.
set -euo pipefail

PORT="${1:-8000}"
MODE="${2:-cloudflare}"
SERVICE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KEY="$(grep -E '^API_KEY=' "$SERVICE_DIR/.env" 2>/dev/null | head -1 | cut -d= -f2- | tr -d '[:space:]' || true)"

if ! curl -fsS --max-time 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "сервис не отвечает на http://localhost:$PORT/health — сначала запустите его (uvicorn / docker compose up -d)" >&2
  exit 1
fi

print_handout() {  # $1 = публичный URL
  echo
  echo "================ публичный адрес ================"
  echo "UI:   $1/"
  echo "API:  $1/api/v1"
  echo "Ключ: ${KEY:-<API_KEY из service/.env>}   (заголовок X-API-Key)"
  echo "curl -X POST $1/api/v1/analyze -H \"X-API-Key: ${KEY:-<ключ>}\" -F \"file=@video_017.mov\""
  echo "curl -H \"X-API-Key: ${KEY:-<ключ>}\" $1/api/v1/jobs/<job_id>"
  echo "проверка снаружи: deploy/healthcheck.sh <видео.mov> $1"
  echo "================================================="
  echo "(адрес действует, пока работает этот процесс; Ctrl+C — остановить)"
  echo
}

case "$MODE" in
  cloudflare)
    command -v cloudflared >/dev/null || { echo "cloudflared не установлен: brew install cloudflared (см. deploy/cloudflared.md)" >&2; exit 1; }
    LOG="$(mktemp -t cloudflared.XXXXXX.log)"
    cloudflared tunnel --url "http://127.0.0.1:$PORT" --no-autoupdate >"$LOG" 2>&1 &
    CF_PID=$!
    trap 'kill $CF_PID 2>/dev/null || true' EXIT INT TERM
    URL=""
    for i in $(seq 1 60); do
      # адрес туннеля — вида https://слово-слово-слово.trycloudflare.com; служебный https://api.trycloudflare.com (в тексте ошибок) не подходит
      URL="$(grep -oE 'https://[a-z0-9]+(-[a-z0-9]+)+\.trycloudflare\.com' "$LOG" | head -1 || true)"
      [ -n "$URL" ] && break
      kill -0 $CF_PID 2>/dev/null || { echo "cloudflared завершился:"; cat "$LOG"; exit 1; }
      sleep 1
    done
    [ -n "$URL" ] || { echo "не дождались адреса trycloudflare.com; лог: $LOG"; cat "$LOG"; exit 1; }
    print_handout "$URL"
    tail -f "$LOG"
    ;;
  ngrok)
    command -v ngrok >/dev/null || { echo "ngrok не установлен (см. deploy/cloudflared.md, раздел 2)" >&2; exit 1; }
    ngrok http "$PORT" --log=stdout --log-format=logfmt >/dev/null 2>&1 &
    NG_PID=$!
    trap 'kill $NG_PID 2>/dev/null || true' EXIT INT TERM
    URL=""
    for i in $(seq 1 30); do
      URL="$(curl -fsS http://127.0.0.1:4040/api/tunnels 2>/dev/null | python3 -c 'import sys,json;t=[x["public_url"] for x in json.load(sys.stdin).get("tunnels",[]) if x["public_url"].startswith("https")];print(t[0] if t else "")' 2>/dev/null || true)"
      [ -n "$URL" ] && break
      kill -0 $NG_PID 2>/dev/null || { echo "ngrok завершился (authtoken задан? ngrok config add-authtoken ...)"; exit 1; }
      sleep 1
    done
    [ -n "$URL" ] || { echo "не дождались адреса ngrok (панель http://127.0.0.1:4040)"; exit 1; }
    print_handout "$URL"
    wait $NG_PID
    ;;
  *)
    echo "режим: cloudflare | ngrok" >&2; exit 2;;
esac
