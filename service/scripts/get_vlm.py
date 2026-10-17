"""Скачивает веса локальной мультимодальной модели (VLM) в service/models/vlm/ через huggingface_hub.

По умолчанию — Qwen/Qwen2.5-VL-7B-Instruct (Apache-2.0, ~16 ГБ в bf16). Докачка после обрыва поддерживается
(snapshot_download пропускает уже скачанные файлы), прогресс по файлам печатается tqdm.

Использование (из папки service/):
  python scripts/get_vlm.py                                   # 7B в models/vlm/
  python scripts/get_vlm.py --model Qwen/Qwen2.5-VL-72B-Instruct --dest D:\\models\\qwen72b
  python scripts/get_vlm.py --check                           # только проверить, что веса на месте

Qwen2.5-VL-3B-Instruct НЕ использовать: лицензия Qwen Research License (некоммерческая), для кейса не подходит.
Запуск модели и сам сервис это не трогает — см. pipeline/vlm.py и docs_vlm_section.md.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SERVICE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DEST = SERVICE_DIR / "models" / "vlm"
DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

# Известные лицензии репозиториев Qwen2.5-VL (по карточкам моделей на HF, проверено 17.10.2026).
KNOWN_LICENSES = {
    "Qwen/Qwen2.5-VL-7B-Instruct": "Apache-2.0",
    "Qwen/Qwen2.5-VL-32B-Instruct": "Apache-2.0",
    "Qwen/Qwen2.5-VL-72B-Instruct": "Qwen License (есть ограничения, см. LICENSE в репозитории)",
    "Qwen/Qwen2.5-VL-3B-Instruct": "Qwen Research License — ТОЛЬКО НЕКОММЕРЧЕСКОЕ, не подходит",
}
NONCOMMERCIAL_MARKERS = ("-3B-",)


def dir_size_bytes(d: Path) -> int:
    return sum(p.stat().st_size for p in d.rglob("*") if p.is_file())


def human(n: float) -> str:
    for unit in ("Б", "КБ", "МБ", "ГБ", "ТБ"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} ПБ"


def weights_ok(d: Path) -> bool:
    return d.is_dir() and (d / "config.json").is_file() and (any(d.glob("*.safetensors")) or any(d.glob("*.bin")))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"репозиторий на Hugging Face (по умолчанию {DEFAULT_MODEL})")
    ap.add_argument("--dest", default=str(DEFAULT_DEST), help=f"куда класть веса (по умолчанию {DEFAULT_DEST})")
    ap.add_argument("--revision", default=None, help="ветка/тег/коммит репозитория (по умолчанию main)")
    ap.add_argument("--check", action="store_true", help="только проверить наличие весов, ничего не качать")
    ap.add_argument(
        "--allow-noncommercial", action="store_true",
        help="разрешить репозиторий с некоммерческой лицензией (например 3B) — для кейса НЕ использовать",
    )
    args = ap.parse_args(argv)

    dest = Path(args.dest).expanduser().resolve()
    model = args.model.strip()

    if args.check:
        if weights_ok(dest):
            print(f"OK: веса на месте в {dest} ({human(dir_size_bytes(dest))})")
            return 0
        print(f"НЕТ: в {dest} нет config.json и *.safetensors — запустите python scripts/get_vlm.py")
        return 1

    if any(m in model for m in NONCOMMERCIAL_MARKERS) and not args.allow_noncommercial:
        print(
            f"ОТКАЗ: {model} распространяется по Qwen Research License (только некоммерческое использование).\n"
            f"Для кейса используйте {DEFAULT_MODEL} (Apache-2.0). Чтобы всё же скачать — --allow-noncommercial.",
            file=sys.stderr,
        )
        return 2

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Нужен пакет huggingface_hub: pip install -r requirements-vlm.txt", file=sys.stderr)
        return 1

    dest.mkdir(parents=True, exist_ok=True)
    print(f"Модель:   {model}")
    print(f"Лицензия: {KNOWN_LICENSES.get(model, 'неизвестна — проверьте карточку модели на HF перед использованием')}")
    print(f"Папка:    {dest}")
    if weights_ok(dest):
        print(f"В папке уже есть веса ({human(dir_size_bytes(dest))}) — докачиваю недостающее/проверяю.")
    print("Загрузка (можно прервать и запустить снова — докачает с места обрыва)...")

    try:
        snapshot_download(
            repo_id=model,
            revision=args.revision,
            local_dir=str(dest),
            # не нужны для инференса; экономим место и время
            ignore_patterns=["*.md", "*.gitattributes", "*.pth", "*.pt", "*.h5", "*.msgpack", "*.onnx"],
            max_workers=4,
        )
    except KeyboardInterrupt:
        print("\nПрервано. Повторный запуск продолжит загрузку.", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"\nОШИБКА загрузки: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("Проверьте интернет, свободное место (нужно ~17 ГБ) и имя репозитория.", file=sys.stderr)
        return 1

    if not weights_ok(dest):
        print(f"Загрузка завершилась, но в {dest} нет config.json / *.safetensors — проверьте репозиторий.", file=sys.stderr)
        return 1

    size = dir_size_bytes(dest)
    print()
    print(f"Готово: {human(size)} в {dest}")
    print(f"Лицензия модели: {KNOWN_LICENSES.get(model, 'см. файл LICENSE в папке с весами')}. "
          "Веса в git не хранятся (models/vlm/ в .gitignore).")
    print("Дальше: в service/.env добавьте VLM_ENABLED=1 (на GPU с 8–12 ГБ — ещё VLM_4BIT=1) и перезапустите сервис.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
