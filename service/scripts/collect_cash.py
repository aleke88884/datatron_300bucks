"""Кеш признака «бледные предметы на столешнице» (pipeline/cashdetect.py) по всем видео датасета.

Пишет service/results/cash/<видео>.json — только числа (рамки, площади, время), без кадров.
Детекции людей берутся из results/perception/<видео>.json (если нет — считаются детектором).

  python scripts/collect_cash.py                  # все видео из ../open/videos
  python scripts/collect_cash.py --names Х-26 Х-4
  python scripts/collect_cash.py --jobs 4
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

SERVICE_DIR = Path(__file__).resolve().parent.parent
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))


def _one(path: str, out_dir: str) -> tuple[str, float, str]:
    from pipeline.cashdetect import compute_cash_track
    from pipeline.perception import cached_perception, perception_from_dict
    from pipeline.video import probe

    t0 = time.time()
    stem = Path(path).stem.replace("X", "Х")
    info = probe(path)
    pj = SERVICE_DIR / "results" / "perception" / f"{stem}.json"
    if pj.exists():
        perc = perception_from_dict(json.loads(pj.read_text(encoding="utf-8")), info)
    else:
        perc = cached_perception(path, info)
    d = compute_cash_track(path, info, perc)
    out = Path(out_dir) / f"{stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    return stem, time.time() - t0, f"{len(d['frames'])} frames"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default=str(SERVICE_DIR.parent / "open" / "videos"))
    ap.add_argument("--out", default=str(SERVICE_DIR / "results" / "cash"))
    ap.add_argument("--names", nargs="*")
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    paths = sorted(Path(args.videos).glob("*.mov"), key=lambda p: int(re.sub(r"\D", "", p.stem) or 0))
    if args.names:
        want = {n.replace("X", "Х") for n in args.names}
        paths = [p for p in paths if p.stem.replace("X", "Х") in want]
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        for stem, dt, msg in ex.map(_one, [str(p) for p in paths], [args.out] * len(paths)):
            print(f"{stem:6} {dt:6.1f} с  {msg}", flush=True)


if __name__ == "__main__":
    main()
