"""Офлайн-оценка правил по всем видео датасета против разметки — без детектора и GPU.

Использует заранее посчитанные признаки из results/ (perception/*.json, states/*.json) через FEATURES_CACHE,
поэтому работает за ~1–2 минуты на любом ноутбуке. Печатает таблицу по видео, матрицу ошибок, ошибку таймкода
и список ложных срабатываний. Используется для подбора порогов правил (scene.py, rules_scene.py).

  python scripts/eval_rules.py                 # все видео
  python scripts/eval_rules.py --names Х-28 Х-29
  python scripts/eval_rules.py --no-frames     # без кадров-доказательств (быстрее)
  python scripts/eval_rules.py --json results/eval.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

SERVICE_DIR = Path(__file__).resolve().parent.parent
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

os.environ.setdefault("FEATURES_CACHE", str(SERVICE_DIR / "results"))

from pipeline import analyze_video  # noqa: E402

sys.path.insert(0, str(SERVICE_DIR / "scripts"))
from collect_features import load_labels, norm_name  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default=str(SERVICE_DIR.parent / "open" / "videos"))
    ap.add_argument("--labels", default=str(SERVICE_DIR.parent / "open" / "open_labels.xlsx"))
    ap.add_argument("--names", nargs="*")
    ap.add_argument("--no-frames", action="store_true")
    ap.add_argument("--json", default=None)
    ap.add_argument("--tol", type=float, default=3.0, help="допуск по таймкоду, с")
    args = ap.parse_args()
    if args.no_frames:
        os.environ["EVIDENCE_FRAMES"] = "0"
    labels = load_labels(Path(args.labels))
    paths = sorted(Path(args.videos).glob("*.mov"), key=lambda p: int(re.sub(r"\D", "", p.stem) or 0))
    if args.names:
        want = {norm_name(n) for n in args.names}
        paths = [p for p in paths if norm_name(p.stem) in want]
    rows = []
    tp = fp = fn = tn = 0
    t_errs = []
    t_all = time.time()
    print(f"{'видео':6} {'разм.':6} {'t':>5}  {'вердикт':12} {'события':40} {'Δt':>6}  {'scene'}")
    for p in paths:
        name = norm_name(p.stem)
        lbl = labels.get(name) or {}
        t0 = time.time()
        try:
            res = analyze_video(str(p))
        except Exception as exc:  # noqa: BLE001
            print(f"{name:6} ОШИБКА {type(exc).__name__}: {exc}")
            rows.append({"name": name, "error": str(exc)})
            continue
        v = res["verdict"]
        evs = [(x["type_code"], x["timecode_sec"], x.get("confidence")) for x in res["violations"]]
        ctxs = [(x["type_code"], x["timecode_sec"]) for x in res["meta"].get("context_events") or []]
        lv = lbl.get("violation")
        ok = None
        if lv is not None:
            if lv and v == "violation":
                tp += 1; ok = "TP"
            elif lv and v != "violation":
                fn += 1; ok = "FN"
            elif (not lv) and v == "violation":
                fp += 1; ok = "FP"
            else:
                tn += 1; ok = "TN"
        dt = ""
        if lv and evs and lbl.get("timecode") is not None:
            best = min(abs(t - lbl["timecode"]) for _, t, _ in evs)
            t_errs.append(best)
            dt = f"{best:5.1f}"
        sc = res["meta"].get("scene") or {}
        scs = f"c={sc.get('cashier_tracks')} v={sc.get('visitor_tracks')} obj={sc.get('desk_obj_frames')} bag={sc.get('bag_frames')} safe={sc.get('safe_open_frames')} scr={sc.get('screen_states')}"
        if res["meta"].get("perception_error"):
            scs = "PERCEPTION ERROR: " + res["meta"]["perception_error"]
        ev_s = ", ".join(f"{c}@{t:.1f}({cf})" for c, t, cf in evs) + (" | ctx " + ",".join(f"{c}@{t:.0f}" for c, t in ctxs) if ctxs else "")
        print(f"{name:6} {('V' if lv else ('-' if lv is False else '?')):6} {str(lbl.get('timecode') or ''):>5}  {(ok or '') + ' ' + v:12} {ev_s[:40]:40} {dt:>6}  {scs}")
        rows.append({"name": name, "label": lbl, "verdict": v, "ok": ok, "violations": res["violations"] and [{k: x[k] for k in ("type_code", "timecode_sec", "confidence", "description")} for x in res["violations"]],
                     "context": ctxs, "scene": sc, "elapsed": round(time.time() - t0, 1)})
    print()
    print(f"TP={tp} FN={fn} FP={fp} TN={tn}   |  recall={tp / max(1, tp + fn):.2f}  FP на чистых={fp}/{fp + tn}")
    if t_errs:
        import numpy as np

        arr = np.asarray(t_errs)
        print(f"ошибка таймкода (по найденным нарушениям): медиана {np.median(arr):.1f} с, среднее {arr.mean():.1f} с, в допуске ≤{args.tol} с: {(arr <= args.tol).sum()}/{len(arr)}")
    fps_ = [r["name"] for r in rows if r.get("ok") == "FP"]
    fns_ = [r["name"] for r in rows if r.get("ok") == "FN"]
    print(f"ложные: {fps_}")
    print(f"пропуски: {fns_}")
    print(f"время: {time.time() - t_all:.0f} с")
    if args.json:
        Path(args.json).write_text(json.dumps({"rows": rows, "tp": tp, "fn": fn, "fp": fp, "tn": tn}, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
