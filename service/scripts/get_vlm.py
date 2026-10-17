"""Скачивает веса локальной мультимодальной модели (VLM) в service/models/vlm/ через huggingface_hub.

По умолчанию — Qwen/Qwen2.5-VL-7B-Instruct (Apache-2.0, ~16 ГБ в bf16). Докачка после обрыва поддерживается
(snapshot_download пропускает уже скачанные файлы), прогресс по файлам печатается tqdm.

Использование (из папки service/):
  python scripts/get_vlm.py                                   # 7B в models/vlm/
  python scripts/get_vlm.py --model Qwen/Qwen2.5-VL-72B-Instruct --dest D:\\models\\qwen72b
  python scripts/get_vlm.py --check                           # только проверить, что веса на месте
  python scripts/get_vlm.py --gguf                            # CPU/Apple Silicon: Qwen3-VL-2B GGUF (~1.5 ГБ) в models/vlm_gguf/

Режим --gguf — для бэкенда llama.cpp (VLM_BACKEND=llamacpp, страховочная сетка pipeline/safety_net.py):
Qwen/Qwen3-VL-2B-Instruct-GGUF (Apache-2.0): Q4_K_M (1.1 ГБ) + mmproj Q8_0 (0.45 ГБ). Нужен бинарник llama-server
(llama.cpp, MIT): macOS — brew install llama.cpp; Windows/Linux — архив релиза github.com/ggml-org/llama.cpp
(положить llama-server в models/vlm_gguf/bin/ или указать VLM_LLAMA_BIN).

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
    "Qwen/Qwen3-VL-2B-Instruct-GGUF": "Apache-2.0",
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


def download_gguf(dest: Path) -> int:
    """Два файла GGUF прямой загрузкой с HF (без huggingface_hub), с докачкой через *.part."""
    import shutil
    import urllib.request

    sys.path.insert(0, str(SERVICE_DIR))
    from pipeline.vlm import GGUF_FILES, GGUF_REPO

    dest.mkdir(parents=True, exist_ok=True)
    print(f"Модель:   {GGUF_REPO} ({KNOWN_LICENSES.get(GGUF_REPO)})")
    print(f"Папка:    {dest}")
    for name in GGUF_FILES:
        out = dest / name
        if out.exists() and out.stat().st_size > 0:
            print(f"  есть: {name} ({human(out.stat().st_size)})")
            continue
        part = out.with_suffix(out.suffix + ".part")
        have = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(f"https://huggingface.co/{GGUF_REPO}/resolve/main/{name}",
                                     headers={"Range": f"bytes={have}-"} if have else {})
        print(f"  загрузка {name}{f' (докачка с {human(have)})' if have else ''}...")
        try:
            with urllib.request.urlopen(req, timeout=60) as r, open(part, "ab" if have and r.status == 206 else "wb") as w:  # noqa: S310
                shutil.copyfileobj(r, w, 1 << 22)
        except Exception as exc:  # noqa: BLE001
            print(f"ОШИБКА загрузки {name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        part.rename(out)
        print(f"  готово: {name} ({human(out.stat().st_size)})")
    print("Дальше: brew install llama.cpp (или бинарник llama-server в models/vlm_gguf/bin/), затем в .env:")
    print("  VLM_ENABLED=1  VLM_BACKEND=llamacpp   и проверка: python scripts/eval_safetynet.py")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"репозиторий на Hugging Face (по умолчанию {DEFAULT_MODEL})")
    ap.add_argument("--dest", default=str(DEFAULT_DEST), help=f"куда класть веса (по умолчанию {DEFAULT_DEST})")
    ap.add_argument("--revision", default=None, help="ветка/тег/коммит репозитория (по умолчанию main)")
    ap.add_argument("--check", action="store_true", help="только проверить наличие весов, ничего не качать")
    ap.add_argument("--gguf", action="store_true", help="скачать GGUF Qwen3-VL-2B для llama.cpp в models/vlm_gguf/")
    ap.add_argument(
        "--allow-noncommercial", action="store_true",
        help="разрешить репозиторий с некоммерческой лицензией (например 3B) — для кейса НЕ использовать",
    )
    args = ap.parse_args(argv)

    if args.gguf:
        gdest = Path(args.dest).expanduser().resolve() if args.dest != str(DEFAULT_DEST) else SERVICE_DIR / "models" / "vlm_gguf"
        return download_gguf(gdest)

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
