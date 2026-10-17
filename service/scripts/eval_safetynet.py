"""Оценка страховочной сетки (pipeline/safety_net.py) на всех видео датасета: ложные на чистых и время.

Сетка запускается на КАЖДОМ видео как будто правила ничего не нашли (SAFETY_NET_ALWAYS=1, стресс-тест): так
проверяется, что она не даёт ложных на 20 чистых роликах, даже когда ей разрешено срабатывать везде. Затем
печатается итог «правила + сетка» в штатном режиме (сетка — только где правила промолчали).

Ворота включения (печатаются в конце): FP на чистых = 0 и время сетки ≤ бюджета на самом длинном видео.
Если ворота не пройдены — на этой машине сетку держать выключенной (SAFETY_NET=0, VLM_ENABLED=0).

Нужна модель (иначе будут только ответы из кеша results/safety_net/*.json):
    VLM_ENABLED=1 VLM_BACKEND=llamacpp python scripts/eval_safetynet.py
    VLM_ENABLED=1 VLM_BACKEND=llamacpp VLM_SERVER_URL=http://127.0.0.1:8977 python scripts/eval_safetynet.py --names Х-40

Ответы модели кешируются в results/safety_net/<видео>.json (только ключ вопроса → true/false/null, без кадров).
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
os.environ["EVIDENCE_FRAMES"] = "0"
os.environ.setdefault("SAFETY_NET", "1")

from pipeline import analyze_video  # noqa: E402

sys.path.insert(0, str(SERVICE_DIR / "scripts"))
from collect_features import load_labels, norm_name  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default=str(SERVICE_DIR.parent / "open" / "videos"))
    ap.add_argument("--labels", default=str(SERVICE_DIR.parent / "open" / "open_labels.xlsx"))
    ap.add_argument("--names", nargs="*")
    ap.add_argument("--json", default=None)
    ap.add_argument("--budget", type=float, default=60.0, help="допустимое время сетки на видео, с")
    args = ap.parse_args()
    labels = load_labels(Path(args.labels))
    paths = sorted(Path(args.videos).glob("*.mov"), key=lambda p: int(re.sub(r"\D", "", p.stem) or 0))
    if args.names:
        want = {norm_name(n) for n in args.names}
        paths = [p for p in paths if norm_name(p.stem) in want]

    rows = []
    stress_fp, stress_tp = [], []
    final = {"TP": 0, "FN": 0, "FP": 0, "TN": 0}
    added_tp, added_fp = [], []
    max_net_sec = 0.0
    t_all = time.time()
    print(f"{'видео':6} {'разм':4} {'t':>5}  {'правила':22} {'сетка (стресс)':34} {'сек':>5}  детали")
    for p in paths:
        name = norm_name(p.stem)
        lbl = labels.get(name) or {}
        lv = lbl.get("violation")
        # 1) стресс: сетка работает на каждом видео
        os.environ["SAFETY_NET_ALWAYS"] = "1"
        res = analyze_video(str(p))
        net = res["meta"].get("safety_net") or {}
        rule_ev = [v for v in res["violations"] if v.get("source") != "safety_net"]
        net_ev = [v for v in res["violations"] if v.get("source") == "safety_net"]
        max_net_sec = max(max_net_sec, float(net.get("elapsed_sec") or 0.0))
        if net_ev and lv is False:
            stress_fp.append(name)
        if net_ev and lv:
            stress_tp.append(name)
        # 2) штатный режим: сетка только там, где правила промолчали
        os.environ["SAFETY_NET_ALWAYS"] = "0"
        res2 = res if rule_ev else analyze_video(str(p))
        v2 = res2["verdict"]
        net2 = [v for v in res2["violations"] if v.get("source") == "safety_net"]
        if lv is not None:
            k = ("TP" if v2 == "violation" else "FN") if lv else ("FP" if v2 == "violation" else "TN")
            final[k] += 1
            if net2 and not rule_ev:
                (added_tp if lv else added_fp).append(name)
        rs = ", ".join(f"{v['type_code']}@{v['timecode_sec']:.0f}" for v in rule_ev) or "-"
        ns = ", ".join(f"{v['type_code']}@{v['timecode_sec']:.1f}" for v in net_ev) or "-"
        det = " ".join(f"{d.get('probe')}@{d.get('t0')}:{'OK' if d.get('ok') else d.get('reason')}" for d in net.get("details") or [])
        print(f"{name:6} {('V' if lv else ('-' if lv is False else '?')):4} {str(lbl.get('timecode') or ''):>5}  {rs[:22]:22} {ns[:34]:34} "
              f"{float(net.get('elapsed_sec') or 0):5.1f}  vlm={net.get('vlm')} calls={net.get('vlm_calls')} {det}")
        rows.append({"name": name, "label": lbl, "rules": rs, "net": [{k: v[k] for k in ("type_code", "timecode_sec", "description")} for v in net_ev],
                     "net_report": net, "final_verdict": v2})
    print()
    print(f"стресс-тест (сетка на всех видео): срабатывания на чистых = {len(stress_fp)} {stress_fp}; на нарушениях = {len(stress_tp)} {stress_tp}")
    print(f"итог правила+сетка: TP={final['TP']} FN={final['FN']} FP={final['FP']} TN={final['TN']}; сетка добавила TP {added_tp}, FP {added_fp}")
    print(f"макс. время сетки на видео: {max_net_sec:.1f} с (бюджет {args.budget:.0f} с); всего {time.time() - t_all:.0f} с")
    gate = not stress_fp and max_net_sec <= args.budget
    print(f"ВОРОТА: {'ПРОЙДЕНЫ — сетку можно включать на этой машине' if gate else 'НЕ ПРОЙДЕНЫ — держать SAFETY_NET=0'}")
    if args.json:
        Path(args.json).write_text(json.dumps({"rows": rows, "stress_fp": stress_fp, "stress_tp": stress_tp, "final": final,
                                               "max_net_sec": max_net_sec, "gate": gate}, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
