#!/usr/bin/env bash
# Проверка сервиса по ТЗ: /health, 401 без ключа, POST /api/v1/analyze с видео, ожидание done, вердикт.
# Использование:
#   deploy/healthcheck.sh <путь к .mov> [BASE_URL]
#   BASE_URL по умолчанию http://localhost:8000; ключ — из переменной API_KEY или из service/.env.
# Примеры:
#   deploy/healthcheck.sh ../open/videos/Х-11.mov
#   API_KEY=... deploy/healthcheck.sh video_017.mov https://xxxx.trycloudflare.com
set -euo pipefail

VIDEO="${1:-}"
BASE="${2:-http://localhost:8000}"
BASE="${BASE%/}"
SERVICE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -z "$VIDEO" ] || [ ! -f "$VIDEO" ]; then
  echo "использование: $0 <путь к .mov> [BASE_URL]" >&2
  exit 2
fi
if [ -z "${API_KEY:-}" ] && [ -f "$SERVICE_DIR/.env" ]; then
  API_KEY="$(grep -E '^API_KEY=' "$SERVICE_DIR/.env" | head -1 | cut -d= -f2- | tr -d '[:space:]' | tr -d '"'"'")"
fi
[ -n "${API_KEY:-}" ] || { echo "нет API_KEY (переменная окружения или service/.env; сгенерировать: python3 deploy/gen_api_key.py)" >&2; exit 2; }

PY=python3
command -v "$PY" >/dev/null || PY=python

FAIL=0
check() { if [ "$1" = "$2" ]; then echo "   ✔ $3"; else echo "   ✘ $3 (ожидали '$2', получили '$1')"; FAIL=1; fi; }

echo "→ GET $BASE/health"
health="$(curl -fsS --max-time 15 "$BASE/health")" || { echo "   ✘ /health недоступен"; exit 1; }
echo "   $health"

echo "→ доступ без ключа / с неверным ключом"
code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 30 -X POST "$BASE/api/v1/analyze" -F "file=@$VIDEO")
check "$code" 401 "POST /api/v1/analyze без ключа → 401"
code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 15 -H "X-API-Key: wrong-key" "$BASE/api/v1/jobs/deadbeef")
check "$code" 401 "GET /api/v1/jobs/{id} с неверным ключом → 401"

echo "→ POST $BASE/api/v1/analyze  file=@$(basename "$VIDEO")"
t0=$(date +%s)
resp="$(curl -fsS --max-time 120 -X POST "$BASE/api/v1/analyze" -H "X-API-Key: $API_KEY" -F "file=@$VIDEO")"
JOB_ID="$(printf '%s' "$resp" | "$PY" -c 'import sys,json;print(json.load(sys.stdin)["job_id"])')"
echo "   job_id=$JOB_ID (за $(( $(date +%s) - t0 )) с)"

echo "→ GET $BASE/api/v1/jobs/$JOB_ID (ждём done, до 5 минут)"
body=""
for i in $(seq 1 300); do
  body="$(curl -fsS --max-time 15 -H "X-API-Key: $API_KEY" "$BASE/api/v1/jobs/$JOB_ID")"
  read -r st stage pct <<<"$("$PY" -c 'import sys,json;j=json.loads(sys.argv[1]);p=j.get("progress") or {};print(j["status"], p.get("stage","-"), p.get("percent","-"))' "$body")"
  if [ "$st" != "processing" ]; then break; fi
  printf '\r   processing %s %s%%   ' "$stage" "$pct"
  sleep 1
done
echo
echo "   обработка заняла $(( $(date +%s) - t0 )) с (с учётом очереди)"

"$PY" - "$body" <<'PYEOF' || FAIL=1
import base64, json, sys
j = json.loads(sys.argv[1])
ok = True
def check(cond, msg):
    global ok
    print(("   ✔ " if cond else "   ✘ ") + msg)
    ok = ok and cond

check(j.get("status") == "done", f"status == done (получили {j.get('status')!r}{': ' + str(j.get('error')) if j.get('error') else ''})")
if j.get("status") == "done":
    check(j.get("verdict") in ("violation", "no_violation"), f"verdict ∈ {{violation, no_violation}}: {j.get('verdict')}")
    v = j.get("violations")
    check(isinstance(v, list), "violations — список")
    check((j.get("verdict") == "violation") == bool(v), "verdict согласован со списком violations")
    for i, e in enumerate(v or []):
        check(all(k in e for k in ("type", "timecode_sec", "description", "evidence_frames")), f"[{i}] поля ТЗ присутствуют")
        check(isinstance(e.get("timecode_sec"), (int, float)), f"[{i}] timecode_sec — число: {e.get('timecode_sec')}")
        fr = e.get("evidence_frames") or []
        check(1 <= len(fr) <= 3, f"[{i}] 1–3 кадра-доказательства ({len(fr)})")
        for f in fr:
            raw = base64.b64decode(f.get("image_base64", "")[:64])
            check(raw[:3] == b"\xff\xd8\xff", f"[{i}] кадр t={f.get('timestamp_sec')} — JPEG base64")
        r = e.get("rule") or {}
        check(all(k in r for k in ("document", "clause", "url")), f"[{i}] rule{{document, clause, url}}")
        print(f"      {e.get('type_code','')} {e['type']} @ {e['timecode_sec']} с — {r.get('document_short') or r.get('clause','')}")
    print(f"   вердикт: {j.get('verdict')}, событий: {len(v or [])}, контекст: {[c.get('type_code') for c in (j.get('meta') or {}).get('context_events', [])]}")
sys.exit(0 if ok else 1)
PYEOF

echo
if [ "$FAIL" = 0 ]; then echo "HEALTHCHECK OK"; else echo "HEALTHCHECK FAILED"; exit 1; fi
