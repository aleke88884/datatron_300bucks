"""Пиксельные признаки правила T8 (личные вещи, напитки) по всем видео → results/personal_items/<видео>.json.

Правило T8 (pipeline/personal_items.py) читает кадры видео только для кандидатов: доля зелёного в рамке сумки
(инкассаторская сумка — не личная вещь) и детекции на увеличенном кропе зоны стола (уточнение момента появления
напитка). При заданном FEATURES_CACHE эти числа запоминаются в results/personal_items/, и офлайн-оценка
(scripts/eval_rules.py) больше не декодирует видео. Скрипт строит сцену из results/perception + results/states,
запускает только T8 и печатает найденные события — заодно это быстрый отчёт по T8.

  python scripts/collect_items.py                  # все видео
  python scripts/collect_items.py --names Х-40 Х-46
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

SERVICE_DIR = Path(__file__).resolve().parent.parent
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))
os.environ.setdefault("FEATURES_CACHE", str(SERVICE_DIR / "results"))

from pipeline.context import VideoContext  # noqa: E402
from pipeline.perception import perception_from_dict  # noqa: E402
from pipeline.personal_items import rule_t8  # noqa: E402
from pipeline.scene import build_scene  # noqa: E402
from pipeline.video import probe  # noqa: E402

sys.path.insert(0, str(SERVICE_DIR / "scripts"))
from collect_features import norm_name  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default=str(SERVICE_DIR.parent / "open" / "videos"))
    ap.add_argument("--names", nargs="*")
    args = ap.parse_args()
    cache = Path(os.environ["FEATURES_CACHE"])
    paths = sorted(Path(args.videos).glob("*.mov"), key=lambda p: int(re.sub(r"\D", "", p.stem) or 0))
    if args.names:
        want = {norm_name(n) for n in args.names}
        paths = [p for p in paths if norm_name(p.stem) in want]
    for p in paths:
        name = norm_name(p.stem)
        pj, sj = cache / "perception" / f"{name}.json", cache / "states" / f"{name}.json"
        if not (pj.exists() and sj.exists()):
            print(f"{name:6} нет perception/states в {cache} — сначала scripts/collect_features.py")
            continue
        info = probe(str(p))
        perc = perception_from_dict(json.loads(pj.read_text(encoding="utf-8")), info)
        scene = build_scene(perc, json.loads(sj.read_text(encoding="utf-8")), info)
        import numpy as np

        ctx = VideoContext(path=str(p), info=info, bright_t=np.zeros(0), bright_v=np.zeros(0))
        ctx.scene = scene
        ctx.cache["perception"] = perc
        evs = rule_t8(ctx)
        s = ", ".join(f"T8@{e.timecode_sec:.1f}({e.extras.get('kind')}, {e.confidence})" for e in evs) or "—"
        print(f"{name:6} {s}")


if __name__ == "__main__":
    main()
