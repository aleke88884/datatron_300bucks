#!/usr/bin/env python3
"""Генерирует ключ API (secrets.token_urlsafe(32)) и записывает/обновляет API_KEY в service/.env.

Использование (из любой папки):
    python3 service/deploy/gen_api_key.py            # новый ключ → .env, печатает ключ
    python3 service/deploy/gen_api_key.py --show     # только показать текущий ключ из .env
    python3 service/deploy/gen_api_key.py --env /путь/к/.env

Если .env нет — создаётся из .env.example (если есть) или с одной строкой API_KEY.
После смены ключа сервис нужно перезапустить.
"""
from __future__ import annotations

import argparse
import re
import secrets
import sys
from pathlib import Path

SERVICE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_ENV = SERVICE_DIR / ".env"
EXAMPLE_ENV = SERVICE_DIR / ".env.example"
KEY_RE = re.compile(r"^\s*API_KEY\s*=.*$", re.M)


def read_key(env_path: Path) -> str | None:
    if not env_path.exists():
        return None
    m = KEY_RE.search(env_path.read_text(encoding="utf-8"))
    if not m:
        return None
    return m.group(0).split("=", 1)[1].strip().strip("\"'")


def write_key(env_path: Path, key: str) -> None:
    if env_path.exists():
        text = env_path.read_text(encoding="utf-8")
    elif EXAMPLE_ENV.exists():
        text = EXAMPLE_ENV.read_text(encoding="utf-8")
    else:
        text = ""
    line = f"API_KEY={key}"
    if KEY_RE.search(text):
        text = KEY_RE.sub(line, text, count=1)
    else:
        if text and not text.endswith("\n"):
            text += "\n"
        text += line + "\n"
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text(text, encoding="utf-8")
    try:
        env_path.chmod(0o600)
    except OSError:
        pass


def main() -> int:
    ap = argparse.ArgumentParser(description="Ключ API для заголовка X-API-Key → service/.env")
    ap.add_argument("--env", type=Path, default=DEFAULT_ENV, help=f"путь к .env (по умолчанию {DEFAULT_ENV})")
    ap.add_argument("--show", action="store_true", help="не генерировать, а показать текущий ключ")
    args = ap.parse_args()

    if args.show:
        key = read_key(args.env)
        if not key or key == "change-me":
            print(f"API_KEY не задан в {args.env}", file=sys.stderr)
            return 1
        print(key)
        return 0

    key = secrets.token_urlsafe(32)
    write_key(args.env, key)
    print(key)
    print(f"записано в {args.env}; перезапустите сервис, чтобы ключ вступил в силу", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
