"""Скачивает веса детектора YOLOX (ONNX, Apache-2.0) и модели позы RTMPose-m (ONNX, Apache-2.0) в service/models/
и проверяет sha256.

Использование: python scripts/get_models.py [--large] [--no-pose]
  по умолчанию — yolox_m.onnx (97 МБ) и rtmpose_m_body7.onnx (52 МБ, для правил по рукам T6),
  с --large дополнительно yolox_l.onnx (207 МБ).
Источники: релиз 0.1.1rc0 репозитория Megvii-BaseDetection/YOLOX (GitHub);
RTMPose-m body7 256x192 — ONNX-сборка OpenMMLab MMPose (download.openmmlab.com, проект rtmposev1).
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
    "rtmpose_m_body7.onnx": "5c0a4bf67953e6d2ac43ce15e77dc9d5d354ae18430a47d2c5963a7bc5683e3c",
}
# модели, которые лежат внутри zip-архива: имя -> (url архива, суффикс файла внутри архива)
ZIPPED = {
    "rtmpose_m_body7.onnx": (
        "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
        "rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip",
        "end2end.onnx",
    ),
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
    url = ZIPPED[name][0] if name in ZIPPED else BASE + name
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
    if name in ZIPPED:
        import zipfile

        with zipfile.ZipFile(tmp) as zf:
            member = next(m for m in zf.namelist() if m.endswith(ZIPPED[name][1]))
            data = zf.read(member)
        tmp.write_bytes(data)
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
    ap.add_argument("--no-pose", action="store_true", help="не скачивать модель позы (правила по рукам будут выключены)")
    args = ap.parse_args()
    names = ["yolox_m.onnx"] + (["yolox_l.onnx"] if args.large else []) + ([] if args.no_pose else ["rtmpose_m_body7.onnx"])
    for n in names:
        fetch(n)


if __name__ == "__main__":
    main()
