"""Настройки сервиса. Читаются из переменных окружения / файла .env рядом с service/."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

SERVICE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(SERVICE_DIR / ".env")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


class Settings:
    API_KEY: str = os.environ.get("API_KEY", "").strip()
    MAX_UPLOAD_MB: int = _env_int("MAX_UPLOAD_MB", 300)
    JOB_TIMEOUT_SEC: int = _env_int("JOB_TIMEOUT_SEC", 240)
    DATA_DIR: Path = Path(os.environ.get("DATA_DIR", str(SERVICE_DIR / "data"))).resolve()
    KEEP_UPLOADS: bool = _env_bool("KEEP_UPLOADS", False)
    ALLOWED_EXTENSIONS: tuple[str, ...] = tuple(
        e.strip().lower() if e.strip().startswith(".") else "." + e.strip().lower()
        for e in os.environ.get("ALLOWED_EXTENSIONS", ".mov,.mp4").split(",")
        if e.strip()
    )
    LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO").upper()

    @property
    def max_upload_bytes(self) -> int:
        return self.MAX_UPLOAD_MB * 1024 * 1024


settings = Settings()
