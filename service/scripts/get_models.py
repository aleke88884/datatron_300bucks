"""Скачивает веса детектора YOLOX (ONNX, Apache-2.0) в service/models/ и проверяет sha256.

Использование: python scripts/get_models.py [--large]
  по умолчанию — yolox_m.onnx (97 МБ), с --large дополнительно yolox_l.onnx (207 МБ).
Источник: релиз 0.1.1rc0 репозитория Megvii-BaseDetection/YOLOX (GitHub).
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
BASE = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/"
MODELS = {
    "yolox_m.onnx": "21ff6cfdeb53b013bac2249599e55f00bff3cfdfdab37ed7a4620818c1d15b3f",
    "yolox_l.onnx": "7860ae79de6c89a3c1eb72ae9a2756c0ccfbe04b7791bb5880afabd97855a411",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name: str) -> None:
    dst = MODELS_DIR / name
    if dst.exists() and sha256(dst) == MODELS[name]:
        print(f"ok   {name} уже на месте ({dst.stat().st_size / 1e6:.0f} МБ), sha256 совпал — скачивать не нужно")
        return
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".part")
    url = BASE + name
    print(f"load {name}: скачиваю {url}")
    import time

    t0 = time.time()

    def hook(blocks: int, block_size: int, total: int) -> None:
        done = min(blocks * block_size, total) if total > 0 else blocks * block_size
        dt = max(time.time() - t0, 1e-3)
        if total > 0:
            sys.stdout.write(f"\r     {100 * done / total:5.1f} %  {done / 1e6:6.1f} / {total / 1e6:.1f} МБ  {done / 1e6 / dt:5.1f} МБ/с   ")
        else:
            sys.stdout.write(f"\r     {done / 1e6:6.1f} МБ  {done / 1e6 / dt:5.1f} МБ/с   ")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, tmp, hook)
    print()
    print(f"     скачано за {time.time() - t0:.0f} с, проверяю sha256 ...")
    got = sha256(tmp)
    if got != MODELS[name]:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"sha256 не совпал для {name}: {got}")
    tmp.replace(dst)
    print(f"ok   {name} сохранён в {dst}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--large", action="store_true", help="дополнительно скачать yolox_l.onnx")
    args = ap.parse_args()
    names = ["yolox_m.onnx"] + (["yolox_l.onnx"] if args.large else [])
    for n in names:
        fetch(n)


if __name__ == "__main__":
    main()
