#!/usr/bin/env bash
# Smoke-тест: поднимает сервер, отправляет Х-11.mov (ожидаем violation с T3 — экран АРМ активен после выключения света — и контекст T9 в meta) и Х-3.mov (ожидаем no_violation),
# печатает JSON (base64 кадров сокращён), проверяет 401 без ключа и схему ответа.
# Использование: scripts/smoke.sh [порт]   (из папки service/ или из корня проекта)
set -euo pipefail

SERVICE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT_DIR="$(cd "$SERVICE_DIR/.." && pwd)"
PORT="${1:-8010}"
BASE="http://127.0.0.1:$PORT"
PY="$SERVICE_DIR/.venv/bin/python"
[ -x "$PY" ] || PY=python3

cd "$SERVICE_DIR"
[ -f .env ] || { echo "нет service/.env — скопируйте .env.example и задайте API_KEY"; exit 1; }
API_KEY="$(grep -E '^API_KEY=' .env | head -1 | cut -d= -f2- | tr -d '[:space:]')"
[ -n "$API_KEY" ] || { echo "API_KEY пуст в .env"; exit 1; }

V_POS="$ROOT_DIR/open/videos/Х-11.mov"
V_NEG="$ROOT_DIR/open/videos/Х-3.mov"
for f in "$V_POS" "$V_NEG"; do [ -f "$f" ] || { echo "нет файла $f"; exit 1; }; done

LOG="$(mktemp -t smoke-server.XXXXXX.log)"
"$PY" -m uvicorn app.main:app --host 127.0.0.1 --port "$PORT" --log-level warning >"$LOG" 2>&1 &
SERVER_PID=$!
cleanup() { kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT

echo "→ ждём /health на $BASE"
for i in $(seq 1 60); do
  if curl -fsS "$BASE/health" >/dev/null 2>&1; then break; fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo "сервер упал:"; cat "$LOG"; exit 1; fi
  sleep 0.5
done
curl -fsS "$BASE/health" >/dev/null || { echo "сервер не поднялся:"; cat "$LOG"; exit 1; }
echo "   health OK"

FAIL=0
check() { if [ "$1" = "$2" ]; then echo "   ✔ $3"; else echo "   ✘ $3 (ожидали '$2', получили '$1')"; FAIL=1; fi; }

echo "→ запросы без ключа / с неверным ключом"
code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE/api/v1/analyze" -F "file=@$V_NEG")
check "$code" 401 "POST /api/v1/analyze без ключа → 401"
code=$(curl -s -o /dev/null -w '%{http_code}' -H "X-API-Key: wrong-key" "$BASE/api/v1/jobs/deadbeef")
check "$code" 401 "GET /api/v1/jobs/{id} с неверным ключом → 401"
code=$(curl -s -o /dev/null -w '%{http_code}' -H "X-API-Key: $API_KEY" "$BASE/api/v1/jobs/deadbeef")
check "$code" 404 "GET несуществующего job → 404"

submit() {  # $1 = путь к видео → job_id
  curl -fsS -X POST "$BASE/api/v1/analyze" -H "X-API-Key: $API_KEY" -F "file=@$1" | "$PY" -c 'import sys,json;print(json.load(sys.stdin)["job_id"])'
}
wait_job() {  # $1 = job_id → печатает финальный JSON
  for i in $(seq 1 600); do
    body=$(curl -fsS -H "X-API-Key: $API_KEY" "$BASE/api/v1/jobs/$1")
    st=$("$PY" -c 'import sys,json;print(json.loads(sys.argv[1])["status"])' "$body")
    if [ "$st" != "processing" ]; then echo "$body"; return 0; fi
    sleep 1
  done
  echo "$body"; return 1
}

echo "→ отправляем $(basename "$V_POS") (ожидаем violation: T3 + контекст T9 в meta.context_events)"
JOB1=$(submit "$V_POS"); echo "   job_id=$JOB1"
echo "→ отправляем $(basename "$V_NEG") (ожидаем no_violation)"
JOB2=$(submit "$V_NEG"); echo "   job_id=$JOB2"

R1=$(wait_job "$JOB1")
R2=$(wait_job "$JOB2")

echo; echo "=== результат $(basename "$V_POS") ==="
"$PY" - "$R1" <<'PYEOF'
import json, sys
j = json.loads(sys.argv[1])
for v in j.get("violations") or []:
    for f in v.get("evidence_frames", []):
        f["image_base64"] = f["image_base64"][:40] + f"...({len(f['image_base64'])} chars)"
print(json.dumps(j, ensure_ascii=False, indent=2))
PYEOF
echo; echo "=== результат $(basename "$V_NEG") ==="
echo "$R2" | "$PY" -c 'import sys,json;print(json.dumps(json.load(sys.stdin),ensure_ascii=False,indent=2))'

echo; echo "→ проверки"
"$PY" - "$R1" "$R2" <<'PYEOF' || FAIL=1
import base64, json, sys
pos, neg = json.loads(sys.argv[1]), json.loads(sys.argv[2])
ok = True
def check(cond, msg):
    global ok
    print(("   ✔ " if cond else "   ✘ ") + msg)
    ok = ok and cond

def validate_schema(j, name):
    check(set(["job_id", "status", "verdict", "violations"]).issubset(j), f"{name}: есть job_id/status/verdict/violations")
    check(j["status"] == "done", f"{name}: status == done")
    check(j["verdict"] in ("no_violation", "violation"), f"{name}: verdict ∈ {{no_violation, violation}}")
    check(isinstance(j["violations"], list), f"{name}: violations — список")
    check((j["verdict"] == "violation") == (len(j["violations"]) > 0), f"{name}: verdict согласован со списком violations")
    for i, v in enumerate(j["violations"]):
        check(all(k in v for k in ("type", "timecode_sec", "description", "evidence_frames")), f"{name}[{i}]: поля ТЗ присутствуют")
        check(isinstance(v["timecode_sec"], (int, float)), f"{name}[{i}]: timecode_sec — число")
        check(1 <= len(v["evidence_frames"]) <= 3, f"{name}[{i}]: 1–3 кадра-доказательства")
        for f in v["evidence_frames"]:
            raw = base64.b64decode(f["image_base64"])
            check(raw[:3] == b"\xff\xd8\xff" and isinstance(f["timestamp_sec"], (int, float)), f"{name}[{i}]: кадр — JPEG base64 с timestamp_sec")
        check("rule" in v and all(k in v["rule"] for k in ("document", "clause", "url")), f"{name}[{i}]: rule{{document,clause,url}}")

validate_schema(pos, "Х-11")
validate_schema(neg, "Х-3")
ctxev = (pos.get("meta") or {}).get("context_events") or []
check(any(e.get("type_code") == "T9" for e in ctxev), "Х-11: в meta.context_events есть T9 (свет выключен)")
check(all(v.get("type_code") != "T9" for v in pos["violations"]), "Х-11: T9 не выдаётся как нарушение (контекст)")
check(pos["verdict"] == "violation" and any(v.get("type_code") == "T3" for v in pos["violations"]), "Х-11 → violation с T3 (экран АРМ активен после выключения света)")
check(all(1 <= len(v["evidence_frames"]) <= 3 for v in pos["violations"]), "Х-11: у каждого нарушения 1–3 кадра-доказательства")
check(neg["verdict"] == "no_violation" and neg["violations"] == [], "Х-3 → no_violation, violations == []")
sys.exit(0 if ok else 1)
PYEOF

echo
if [ "$FAIL" = 0 ]; then echo "SMOKE OK"; else echo "SMOKE FAILED"; echo "--- лог сервера ---"; cat "$LOG"; exit 1; fi
