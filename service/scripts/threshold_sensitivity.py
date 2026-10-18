"""Устойчивость порогов правил (ПЛАН §14: «независимой проверки порогов на устойчивость (±20 %) нет»).

Каждый порог уровня правил/сцены по очереди сдвигается на −20 % и +20 % (целые — с округлением, не меньше 1), остальные
остаются как есть; по кешу признаков (results/, FEATURES_CACHE) прогоняются все 42 видео и считается матрица ошибок против
open/open_labels.xlsx. Пороги признаков, которые лежат в кеше (детектор купюр, область двери, предметы), офлайн не
проверить — для них нужен живой прогон.

    python scripts/threshold_sensitivity.py                      # все пороги, 4 процесса
    python scripts/threshold_sensitivity.py --only rules_door     # один модуль
    python scripts/threshold_sensitivity.py --out results/threshold_sensitivity.json --resume

Итог: таблица порогов, при сдвиге которых меняется TP или появляются ложные (FP), — «хрупкие» пороги.
"""
from __future__ import annotations

import argparse
import importlib
import json
import multiprocessing as mp
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

SERVICE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SERVICE_DIR))
sys.path.insert(0, str(SERVICE_DIR / "scripts"))
os.environ.setdefault("FEATURES_CACHE", str(SERVICE_DIR / "results"))
os.environ["EVIDENCE_FRAMES"] = "0"

PARAMS = {
    "pipeline.rules_scene": ["T1_AWAY_SEC", "T1_PRESENT_BEFORE_SEC", "T3_AWAY_SEC", "T4_SEC", "T5_SEC", "T5_AT_DESK_FRAC",
                             "T6_AFTER_SEC", "T6_MIN_FRAC", "T7_BEFORE_SEC", "T7_MIN_SAT", "T7_MIN_AREA", "T7_MIN_VAL",
                             "T7_CASH_GAP_SEC", "T8_DRINK_MIN_DET", "T8_DRINK_MIN_SCORE", "T8_BAG_MIN_DET", "T8_BAG_MIN_SCORE",
                             "T10_VISITOR_SEC", "T10_WINDOW_SEC", "T13_EMPTY_SEC"],
    "pipeline.rules_door": ["OPEN_RATIO", "OPEN_MIN", "OPEN_HOLD_SEC", "MIN_VALID_FRAC", "CROSS_MIN_COVER"],
    "pipeline.rules_hands": ["STAND_RATIO", "STAND_KEEP_RATIO", "SEATED_SEC", "SEATED_CV", "T6H_WINDOW_SEC",
                             "T6H_ENGAGED_DIST", "T6H_ENGAGED_SEC"],
    "pipeline.scene": ["STRONG_SCORE", "CASHIER_MIN_WORK_FRAC", "DESK_PAD", "BAG_AREA_MIN", "DESK_OBJ_MIN", "LIGHTS_OFF_RATIO"],
}
FACTORS = (0.8, 1.2)
# константы, которые другие модули импортируют по имени (from .scene import DESK_OBJ_MIN) — патчим и там
ALSO = {("pipeline.scene", "DESK_OBJ_MIN"): ["pipeline.rules_scene", "pipeline.timeline"]}

_VIDEOS: list[Path] = []
_LABELS: dict = {}


# Время в отчётах — по Казахстану (UTC+05:00, единый пояс с 01.03.2024, без летнего времени), с явным поясом в строке:
# если часы сервера идут в UTC, в JSON не будет времени на 5 часов раньше местного.
REPORT_TZ = timezone(timedelta(hours=5))


def report_time(ts: float | None = None) -> str:
    """'2026-10-18 08:47:46+05:00' для момента ts (по умолчанию — сейчас)."""
    return datetime.fromtimestamp(time.time() if ts is None else ts, REPORT_TZ).isoformat(sep=" ", timespec="seconds")


def _init(videos: list[str], labels_path: str) -> None:
    global _VIDEOS, _LABELS
    from collect_features import load_labels  # noqa: E402
    _VIDEOS = [Path(v) for v in videos]
    _LABELS = load_labels(Path(labels_path))
    import pipeline  # noqa: F401 — прогрев импорта в процессе


def _evaluate() -> dict:
    from collect_features import norm_name
    from pipeline import analyze_video
    tp = fn = fp = tn = 0
    per = {}
    for p in _VIDEOS:
        name = norm_name(p.stem)
        lv = (_LABELS.get(name) or {}).get("violation")
        try:
            v = analyze_video(str(p))["verdict"]
        except Exception as exc:  # noqa: BLE001
            per[name] = f"error: {exc}"
            continue
        per[name] = v
        if lv is None:
            continue
        if lv and v == "violation":
            tp += 1
        elif lv:
            fn += 1
        elif v == "violation":
            fp += 1
        else:
            tn += 1
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn, "per": per}


def _task(job: tuple[str, str, float]) -> dict:
    mod_name, attr, factor = job
    t0 = time.time()
    if not attr:
        res = _evaluate()
        res.update(module="", param="baseline", factor=1.0, base=None, value=None)
    else:
        mod = importlib.import_module(mod_name)
        base = getattr(mod, attr)
        val = max(1, round(base * factor)) if isinstance(base, int) and not isinstance(base, bool) else round(base * factor, 4)
        targets = [mod] + [importlib.import_module(m) for m in ALSO.get((mod_name, attr), [])]
        for m in targets:
            setattr(m, attr, val)
        try:
            res = _evaluate()
        finally:
            for m in targets:
                setattr(m, attr, base)
        res.update(module=mod_name, param=attr, factor=factor, base=base, value=val)
    res["sec"] = round(time.time() - t0, 1)
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", default=str(SERVICE_DIR.parent / "open" / "videos"))
    ap.add_argument("--labels", default=str(SERVICE_DIR.parent / "open" / "open_labels.xlsx"))
    ap.add_argument("--only", nargs="*", help="модули (rules_scene, rules_door, rules_hands, scene) или имена порогов")
    ap.add_argument("--procs", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--out", default=str(SERVICE_DIR / "results" / "threshold_sensitivity.json"))
    ap.add_argument("--resume", action="store_true", help="пропустить уже посчитанные варианты из --out")
    ap.add_argument("--max-jobs", type=int, default=0, help="посчитать не больше N вариантов за запуск (0 — все)")
    a = ap.parse_args()

    videos = sorted(str(p) for p in Path(a.videos).glob("*.mov"))
    done: list[dict] = []
    if a.resume and Path(a.out).exists():
        done = json.loads(Path(a.out).read_text(encoding="utf-8")).get("runs", [])
    have = {(r["module"], r["param"], r["factor"]) for r in done}

    jobs: list[tuple[str, str, float]] = []
    if ("", "baseline", 1.0) not in have:
        jobs.append(("", "", 1.0))
    for mod, names in PARAMS.items():
        for n in names:
            short = mod.split(".")[-1]
            if a.only and short not in a.only and n not in a.only:
                continue
            for f in FACTORS:
                if (mod, n, f) not in have:
                    jobs.append((mod, n, f))
    if a.max_jobs:
        jobs = jobs[: a.max_jobs]
    print(f"видео: {len(videos)}; вариантов к расчёту: {len(jobs)}; процессов: {a.procs}", flush=True)

    def save() -> None:
        Path(a.out).write_text(json.dumps({"updated": report_time(), "runs": done},
                                          ensure_ascii=False, indent=1), encoding="utf-8")

    if jobs:
        ctx = mp.get_context("fork" if sys.platform.startswith("linux") else "spawn")
        with ctx.Pool(a.procs, initializer=_init, initargs=(videos, a.labels)) as pool:
            for r in pool.imap_unordered(_task, jobs):
                done.append(r)
                save()
                print(f"  {r['param']:<24} ×{r['factor']:<4} TP {r['tp']:>2} FN {r['fn']:>2} FP {r['fp']:>2} TN {r['tn']:>2}  ({r['sec']} с)",
                      flush=True)

    base = next((r for r in done if r["param"] == "baseline"), None)
    if base:
        print(f"\nбазовая линия: TP {base['tp']} / FN {base['fn']} / FP {base['fp']} / TN {base['tn']}")
        fragile = []
        for r in sorted(done, key=lambda r: (r["module"], r["param"], r["factor"])):
            if r["param"] == "baseline":
                continue
            if (r["tp"], r["fp"]) != (base["tp"], base["fp"]):
                changed = [n for n, v in r["per"].items() if v != base["per"].get(n)]
                fragile.append((r, changed))
        print(f"вариантов посчитано: {len(done) - 1}; меняют результат: {len(fragile)}")
        for r, changed in fragile:
            print(f"  {r['module'].split('.')[-1]}.{r['param']} = {r['value']} (было {r['base']}, ×{r['factor']}): "
                  f"TP {r['tp']} FP {r['fp']} — {', '.join(changed)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
