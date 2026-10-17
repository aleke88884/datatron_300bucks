"""Пересчёт признаков двери (pipeline/door.py) по всем видео → results/door/<видео>.json (только числа).

Детекции людей берутся из results/perception/<видео>.json (их считает scripts/collect_features.py); видео
декодируется заново (4 к/с, уменьшенные серые кадры, в память — на диск кадры не пишутся).

  python scripts/collect_door.py                  # все видео, только отсутствующие
  python scripts/collect_door.py --force          # пересчитать все
  python scripts/collect_door.py --names Х-31 Х-32 --verbose
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from multiprocessing import Pool
from pathlib import Path

SERVICE_DIR = Path(__file__).resolve().parent.parent
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

RESULTS = SERVICE_DIR / "results"


def _one(args: tuple[str, bool]) -> str:
    path, force = args
    from pipeline.door import door_features
    from pipeline.perception import perception_from_dict
    from pipeline.video import probe

    os.environ["FEATURES_CACHE"] = str(RESULTS)
    stem = Path(path).stem.replace("X", "Х")
    out = RESULTS / "door" / f"{stem}.json"
    pj = RESULTS / "perception" / f"{stem}.json"
    if not pj.exists():
        return f"{stem}: нет {pj}"
    if force and out.exists():
        out.unlink()
    t0 = time.time()
    info = probe(path)
    perc = perception_from_dict(json.loads(pj.read_text(encoding="utf-8")), info)
    d = door_features(path, info, perc)
    if d.get("found"):
        return f"{stem}: roi={d['roi']} first_motion={d['t_first_motion']} peak={d['peak']} ({time.time() - t0:.1f} с)"
    return f"{stem}: дверь не найдена — {d.get('reason')} ({time.time() - t0:.1f} с)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default=str(SERVICE_DIR.parent / "open" / "videos"))
    ap.add_argument("--names", nargs="*")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    paths = sorted(Path(args.videos).glob("*.mov"), key=lambda p: int(re.sub(r"\D", "", p.stem) or 0))
    if args.names:
        want = {n.replace("X", "Х") for n in args.names}
        paths = [p for p in paths if p.stem.replace("X", "Х") in want]
    (RESULTS / "door").mkdir(parents=True, exist_ok=True)
    jobs = [(str(p), args.force) for p in paths]
    if args.jobs > 1 and len(jobs) > 1:
        with Pool(args.jobs) as pool:
            for line in pool.imap(_one, jobs):
                print(line, flush=True)
    else:
        for j in jobs:
            print(_one(j), flush=True)


if __name__ == "__main__":
    main()
