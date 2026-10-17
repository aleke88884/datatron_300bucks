"""Поза людей (RTMPose, 17 точек) по всем видео датасета → results/hands/<видео>.json (только числа, без кадров).

Рамки людей берутся из results/perception/<видео>.json (если есть) — поза считается для тех же моментов
времени и тех же людей, что видит пайплайн. Если кеша восприятия нет — восприятие считается по видео.
Кеш нужен только для быстрой офлайн-оценки (scripts/eval_rules.py, FEATURES_CACHE); живой путь сервиса
считает позу по видео сам (pipeline.pose.load_or_compute).

  python scripts/collect_hands.py                     # все видео из ../open/videos
  python scripts/collect_hands.py --names Х-34 Х-19   # только указанные
  python scripts/collect_hands.py --force             # пересчитать, даже если кеш есть
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

SERVICE_DIR = Path(__file__).resolve().parent.parent
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

from pipeline.perception import cached_perception, perception_from_dict  # noqa: E402
from pipeline.pose import POSE_VERSION, run_poses  # noqa: E402
from pipeline.video import probe  # noqa: E402

sys.path.insert(0, str(SERVICE_DIR / "scripts"))
from collect_features import norm_name  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default=str(SERVICE_DIR.parent / "open" / "videos"))
    ap.add_argument("--out", default=str(SERVICE_DIR / "results"))
    ap.add_argument("--names", nargs="*")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "hands").mkdir(parents=True, exist_ok=True)
    paths = sorted(Path(args.videos).glob("*.mov"), key=lambda p: int(re.sub(r"\D", "", p.stem) or 0))
    if args.names:
        want = {norm_name(n) for n in args.names}
        paths = [p for p in paths if norm_name(p.stem) in want]
    t_all = time.time()
    for p in paths:
        name = norm_name(p.stem)
        dst = out / "hands" / f"{name}.json"
        if dst.exists() and not args.force:
            try:
                if json.loads(dst.read_text(encoding="utf-8")).get("version") == POSE_VERSION:
                    print(f"{name:6} кеш есть — пропуск")
                    continue
            except Exception:  # noqa: BLE001
                pass
        t0 = time.time()
        info = probe(str(p))
        pj = out / "perception" / f"{name}.json"
        if pj.exists():
            perc = perception_from_dict(json.loads(pj.read_text(encoding="utf-8")), info)
        else:
            perc = cached_perception(str(p), info)
        d = run_poses(str(p), info, perc)
        dst.write_text(json.dumps(d, separators=(",", ":")), encoding="utf-8")
        n = sum(len(x) for x in d["poses"])
        print(f"{name:6} {info.duration_sec:6.1f} с видео, поз: {n:4d}, {time.time() - t0:5.1f} с")
    print(f"всего: {time.time() - t_all:.0f} с")


if __name__ == "__main__":
    main()
