"""Пакетная проверка живого сервиса (ПЛАН §14 п. 2): несколько видео отправляются ОДНОВРЕМЕННО, как это сделает жюри.

Для каждого задания печатает время от POST до готового ответа, вердикт, совпадение с разметкой и признаки деградации
(meta.degraded, meta.degraded_reason, meta.analysis_complete). Ответ позже 300 с (лимит ТЗ) или тихий `no_violation`
на ролике с нарушением при неполном анализе считаются провалом.

Использование (сервер уже запущен):
    python scripts/batch_check.py --base http://127.0.0.1:8000 Х-47.mov Х-13.mov Х-14.mov
    python scripts/batch_check.py --base https://<host> --out results/batch_check.json Х-47.mov Х-13.mov Х-11.mov

Ключ берётся из --key, переменной API_KEY или service/.env (в командной строке не печатается).
Имена видео ищутся в open/videos/; можно передать и полный путь.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import sys
import time
from pathlib import Path

import requests

SERVICE_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = SERVICE_DIR.parent
VIDEOS_DIR = ROOT_DIR / "open" / "videos"
LABELS_XLSX = ROOT_DIR / "open" / "open_labels.xlsx"
TZ_LIMIT_SEC = 300.0


def read_key(cli_key: str | None) -> str:
    if cli_key:
        return cli_key
    if os.environ.get("API_KEY"):
        return os.environ["API_KEY"].strip()
    env = SERVICE_DIR / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("API_KEY="):
                return line.split("=", 1)[1].strip()
    sys.exit("нет ключа: передайте --key, задайте API_KEY или создайте service/.env")


def resolve_video(name: str) -> Path:
    p = Path(name)
    if p.exists():
        return p
    for cand in (name, name.replace("X", "Х"), name.replace("Х", "X")):
        q = VIDEOS_DIR / cand
        if q.exists():
            return q
    sys.exit(f"не найдено видео {name} (искал в {VIDEOS_DIR})")


def load_labels() -> dict[str, str]:
    """stem (Х-NN, кириллица) -> 'violation' / 'no_violation' по open_labels.xlsx; пусто, если файла или openpyxl нет."""
    try:
        import openpyxl
    except ImportError:
        return {}
    if not LABELS_XLSX.exists():
        return {}
    ws = openpyxl.load_workbook(LABELS_XLSX, read_only=True).active
    rows = list(ws.iter_rows(values_only=True))
    if not rows:
        return {}
    head = [str(c or "").strip().lower() for c in rows[0]]
    name_i = next((i for i, h in enumerate(head) if any(k in h for k in ("video", "видео", "file", "файл", "name"))), 0)
    verd_i = next((i for i, h in enumerate(head) if any(k in h for k in ("verdict", "вердикт", "label", "наруш"))), None)
    out: dict[str, str] = {}
    for r in rows[1:]:
        if not r or r[name_i] is None or verd_i is None or r[verd_i] is None:
            continue
        stem = Path(str(r[name_i]).strip()).stem.replace("X", "Х")
        v = str(r[verd_i]).strip().lower()
        out[stem] = "no_violation" if v.startswith(("no", "без")) or v in ("0", "нет", "false") else "violation"
    return out


def run_one(base: str, key: str, path: Path, t0: float, poll: float, give_up: float) -> dict:
    with open(path, "rb") as f:
        r = requests.post(f"{base}/api/v1/analyze", headers={"X-API-Key": key},
                          files={"file": (path.name, f, "video/quicktime")}, timeout=180)
    posted = time.time()
    rec: dict = {"video": path.name, "post_status": r.status_code, "post_sec": round(posted - t0, 2)}
    if r.status_code != 200:
        rec.update(status="post_failed", error=r.text[:300])
        return rec
    job_id = r.json()["job_id"]
    rec["job_id"] = job_id
    stages: list[str] = []
    while True:
        time.sleep(poll)
        try:
            j = requests.get(f"{base}/api/v1/jobs/{job_id}", headers={"X-API-Key": key}, timeout=30).json()
        except requests.RequestException as exc:
            if time.time() - posted > give_up:
                rec.update(status="timeout", error=str(exc)[:200])
                return rec
            continue
        st = (j.get("progress") or {}).get("stage")
        if st and (not stages or stages[-1] != st):
            stages.append(st)
        if j.get("status") in ("done", "failed"):
            break
        if time.time() - posted > give_up:
            rec.update(status="timeout")
            return rec
    res = j   # ответ GET /jobs/{id} плоский: verdict, violations, meta на верхнем уровне (app/schemas.py JobResponse)
    meta = j.get("meta") or {}
    rec.update(
        status=j["status"],
        total_sec=round(time.time() - posted, 1),
        verdict=res.get("verdict"),
        types=sorted({v.get("type_code") for v in res.get("violations") or []}),
        timecodes=[v.get("timecode_sec") for v in res.get("violations") or []],
        degraded=meta.get("degraded"),
        degraded_reason=meta.get("degraded_reason"),
        analysis_complete=meta.get("analysis_complete"),
        queue_wait_sec=meta.get("queue_wait_sec"),
        pipeline_time_sec=meta.get("pipeline_time_sec"),
        stages=stages,
        error=j.get("error"),
    )
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--key")
    ap.add_argument("--poll", type=float, default=1.0)
    ap.add_argument("--give-up", type=float, default=330.0, help="сколько ждать одно задание, с")
    ap.add_argument("--out", help="куда записать JSON с результатами")
    ap.add_argument("--sequential", action="store_true",
                    help="по одному видео (для собственных роликов, docs/OWN_VIDEOS.md), а не пачкой")
    a = ap.parse_args()

    base = a.base.rstrip("/")
    key = read_key(a.key)
    paths = [resolve_video(v) for v in a.videos]
    labels = load_labels()
    health = requests.get(f"{base}/health", timeout=15).json()
    print(f"сервер {base}: workers={health.get('workers')}, job_timeout={health.get('job_timeout_sec')} с; "
          f"отправляю {'по одному' if a.sequential else 'одновременно'} {len(paths)} видео")

    t0 = time.time()
    if a.sequential:
        recs = []
        for p in paths:
            recs.append(run_one(base, key, p, time.time(), a.poll, a.give_up))
    else:
        with cf.ThreadPoolExecutor(len(paths)) as ex:
            recs = list(ex.map(lambda p: run_one(base, key, p, t0, a.poll, a.give_up), paths))

    fails = 0
    print(f"\n{'видео':<10} {'статус':<8} {'время,с':>8} {'очередь,с':>9}  {'вердикт':<13} {'разметка':<13} {'полный':<6} деградация")
    for rec in recs:
        stem = Path(rec["video"]).stem.replace("X", "Х")
        expected = labels.get(stem)
        rec["expected"] = expected
        problems = []
        if rec.get("status") != "done":
            problems.append(f"статус {rec.get('status')}")
        if (rec.get("total_sec") or 1e9) + rec.get("post_sec", 0) > TZ_LIMIT_SEC:
            problems.append("дольше 5 минут")
        if rec.get("degraded") and not rec.get("degraded_reason"):
            problems.append("деградация без причины")
        if expected == "violation" and rec.get("verdict") == "no_violation" and rec.get("analysis_complete") is False:
            problems.append("тихий no_violation при неполном анализе")
        rec["problems"] = problems
        fails += bool(problems)
        print(f"{rec['video']:<10} {rec.get('status', '?'):<8} {rec.get('total_sec', 0):>8} {rec.get('queue_wait_sec') or 0:>9}  "
              f"{str(rec.get('verdict')):<13} {str(expected):<13} {str(rec.get('analysis_complete')):<6} "
              f"{rec.get('degraded_reason') or '—'}" + (f"   ✘ {', '.join(problems)}" if problems else ""))
    print(f"\nвсего {round(time.time() - t0, 1)} с; провалов: {fails}")

    if a.out:
        out = {"base": base, "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)), "health": health, "jobs": recs}
        Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"записано: {a.out}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
