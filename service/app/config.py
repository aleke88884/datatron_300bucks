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


def available_cpus() -> int:
    """Ядра, реально доступные процессу: affinity и квота cgroup (Docker --cpus), а не все ядра хоста."""
    try:
        n = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        n = os.cpu_count() or 1
    try:
        quota = None
        cpu_max = Path("/sys/fs/cgroup/cpu.max")
        if cpu_max.exists():
            q, per = cpu_max.read_text().split()[:2]
            if q != "max":
                quota = int(q) / int(per)
        else:
            q = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
            per = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
            if q > 0:
                quota = q / per
        if quota:
            n = min(n, max(1, int(quota + 0.5)))
    except (OSError, ValueError):
        pass
    return max(1, n)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


_DEFAULT_EXTENSIONS = ".mov,.mp4,.m4v,.avi,.mkv,.webm,.mpg,.mpeg,.3gp,.mts,.m2ts"


# Значения-заглушки из примеров: сервис с таким ключом не стартует (ключ из .env.example известен всем).
PLACEHOLDER_API_KEYS = frozenset({
    "change-me", "changeme", "change_me", "<ключ>", "<key>", "your-api-key", "your_api_key", "api-key", "secret", "test", "key",
})

# ТЗ: видео, не обработанное за 5 минут, считается необработанным. Жёсткий дедлайн задания никогда не выходит за
# TZ_LIMIT_SEC - TZ_MARGIN_SEC от POST, даже если задание долго стояло в очереди.
TZ_LIMIT_SEC = 300.0
TZ_MARGIN_SEC = 15.0


class Settings:
    API_KEY: str = os.environ.get("API_KEY", "").strip()
    MAX_UPLOAD_MB: int = _env_int("MAX_UPLOAD_MB", 300)
    # Жёсткий лимит на задание (с момента POST): ТЗ даёт 5 минут, оставляем запас на опрос жюри и выдачу результата.
    JOB_TIMEOUT_SEC: int = _env_int("JOB_TIMEOUT_SEC", 270)
    # Мягкий дедлайн (с момента POST): к нему восприятие должно закончиться, дальше пайплайн только деградирует
    # (меньше кадров, без VLM). По умолчанию 80 % от жёсткого: детектор получает 70 % времени до мягкого дедлайна
    # (pipeline/budget.py), после него ещё наличные/дверь/поза/правила/кадры — ~25 с на длинном видео.
    SOFT_DEADLINE_SEC: float = _env_float("SOFT_DEADLINE_SEC", 0.0) or round(0.8 * _env_int("JOB_TIMEOUT_SEC", 270), 1)
    # Минимум рабочего времени, которое получает задание, даже если оно долго стояло в очереди
    MIN_WORK_SEC: float = _env_float("MIN_WORK_SEC", 0.0)   # 0 = 10 % от JOB_TIMEOUT_SEC
    # Число параллельных процессов-воркеров; 0 = авто (1 процесс со всеми ядрами; 2 при ≥ 12 ядрах)
    WORKERS: int = _env_int("WORKERS", 0)
    # Порядок очереди: короткие записи первыми (при пачке загрузок больше видео успевает пройти полный анализ),
    # но задание, прождавшее дольше QUEUE_AGING_SEC, берётся раньше новых.
    QUEUE_AGING_SEC: float = _env_float("QUEUE_AGING_SEC", 45.0)
    # Веб-интерфейс без ключа (ПЛАН: «без авторизации»): страница / ставит HttpOnly-cookie, которой UI вызывает API.
    # Программные клиенты по-прежнему обязаны передавать X-API-Key. UI_OPEN=false — ключ нужен и в интерфейсе.
    UI_OPEN: bool = _env_bool("UI_OPEN", True)
    JOB_TTL_SEC: int = _env_int("JOB_TTL_SEC", 3600)     # готовые задания старше этого срока удаляются из памяти
    DATA_DIR: Path = Path(os.environ.get("DATA_DIR", str(SERVICE_DIR / "data"))).resolve()
    KEEP_UPLOADS: bool = _env_bool("KEEP_UPLOADS", False)
    ALLOWED_EXTENSIONS: tuple[str, ...] = tuple(
        e.strip().lower() if e.strip().startswith(".") else "." + e.strip().lower()
        for e in os.environ.get("ALLOWED_EXTENSIONS", _DEFAULT_EXTENSIONS).split(",")
        if e.strip()
    )
    LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO").upper()

    def api_key_problem(self) -> str | None:
        """Причина, по которой ключ API непригоден (None — ключ в порядке)."""
        if not self.API_KEY:
            return "API_KEY не задан. Создайте service/.env (см. .env.example) и укажите API_KEY: python3 deploy/gen_api_key.py"
        if self.API_KEY.lower() in PLACEHOLDER_API_KEYS:
            return (f"API_KEY={self.API_KEY!r} — значение-заглушка из примера, такой ключ известен всем. "
                    "Сгенерируйте настоящий: python3 deploy/gen_api_key.py (или python3 -c 'import secrets;print(secrets.token_urlsafe(32))')")
        return None

    @property
    def worker_count(self) -> int:
        if self.WORKERS > 0:
            return self.WORKERS
        # Замер на M3 8 ГБ (Х-14): 1 процесс × 8 потоков — 165 с, 2 × 4 потока параллельно — по 294 с каждое.
        # Пропускная способность почти одинакова, а время до первого готового ответа вдвое меньше у одного процесса;
        # очередь с планированием (app/worker.py) тогда сжимает задания равномерно. 2 процесса — только на ≥ 12 ядрах.
        return 2 if available_cpus() >= 12 else 1

    @property
    def threads_per_worker(self) -> int:
        return max(1, available_cpus() // self.worker_count)

    @property
    def max_upload_bytes(self) -> int:
        return self.MAX_UPLOAD_MB * 1024 * 1024


settings = Settings()
