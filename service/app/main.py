"""FastAPI-приложение: API по разделу 10 ТЗ + статический фронтенд."""
from __future__ import annotations

import logging
import os
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.concurrency import run_in_threadpool
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles

from .config import SERVICE_DIR, settings
from .schemas import JobCreated, JobResponse
from .store import Job, JobStore
from .worker import AnalysisWorker

logging.basicConfig(level=settings.LOG_LEVEL, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
log = logging.getLogger("api")

if not settings.API_KEY:
    raise RuntimeError("API_KEY не задан. Создайте service/.env (см. .env.example) и укажите API_KEY.")

STATIC_DIR = SERVICE_DIR / "static"
UPLOAD_DIR = settings.DATA_DIR / "uploads"

store = JobStore()
worker = AnalysisWorker(store)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(key: str | None = Depends(api_key_header)) -> None:
    if not key or not secrets.compare_digest(key, settings.API_KEY):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid or missing X-API-Key")


def purge_stale_uploads() -> int:
    """Остатки загрузок после аварийного перезапуска: тестовые видео не храним."""
    n = 0
    if settings.KEEP_UPLOADS:
        return n
    for p in UPLOAD_DIR.glob("*"):
        try:
            if p.is_file():
                p.unlink()
                n += 1
        except OSError:
            pass
    return n


@asynccontextmanager
async def lifespan(_: FastAPI):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    purged = purge_stale_uploads()
    worker.start()
    log.info(
        "service started; uploads=%s (purged %d), workers=%d x %d threads, soft=%ss hard=%ss, max_upload=%sMB",
        UPLOAD_DIR, purged, worker.n_workers, settings.threads_per_worker, settings.SOFT_DEADLINE_SEC, settings.JOB_TIMEOUT_SEC, settings.MAX_UPLOAD_MB,
    )
    yield
    worker.stop()


app = FastAPI(
    title="ForteBank Cash Desk Compliance",
    description="Локальный сервис анализа видео кассы на нарушения кассовой дисциплины (Datathon 26).",
    version="0.1.0",
    lifespan=lifespan,
)


class _TooLarge(Exception):
    pass


def _save_upload(src, dest: Path, limit: int) -> int:
    size = 0
    with open(dest, "wb") as out:
        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > limit:
                raise _TooLarge()
            out.write(chunk)
    return size


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "time": time.time(),
        "worker_alive": worker.child_alive,
        "workers": {"configured": worker.n_workers, "alive": worker.workers_alive},
        "queue": worker.queue_size,
        "jobs": store.stats(),
    }


@app.post("/api/v1/analyze", response_model=JobCreated, status_code=200, dependencies=[Depends(require_api_key)])
async def analyze(request: Request, file: UploadFile = File(...)) -> JobCreated:
    created_at = time.time()   # 5 минут ТЗ отсчитываются от POST
    # Имя файла может быть любым (кириллица, пробелы, обратные слэши Windows): на диске файл называется по job_id.
    filename = (file.filename or "video.mov").replace("\\", "/").rsplit("/", 1)[-1] or "video.mov"
    ext = Path(filename).suffix.lower()
    if not ext:
        ext = ".mov"            # без расширения: содержимое проверит декодер, а не имя
    elif ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail=f"unsupported file type '{ext}', expected one of {list(settings.ALLOWED_EXTENSIONS)}")

    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > settings.max_upload_bytes + 1_000_000:
        raise HTTPException(status_code=413, detail=f"file too large, limit {settings.MAX_UPLOAD_MB} MB")

    job_id = store.new_id()
    dest = UPLOAD_DIR / f"{job_id}{ext}"
    try:
        size = await run_in_threadpool(_save_upload, file.file, dest, settings.max_upload_bytes)
    except _TooLarge:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=413, detail=f"file too large, limit {settings.MAX_UPLOAD_MB} MB")
    except OSError as exc:
        dest.unlink(missing_ok=True)
        log.error("cannot save upload: %s", exc)
        raise HTTPException(status_code=507, detail="cannot store upload")
    finally:
        await file.close()

    if size == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="empty file")

    job = Job(job_id=job_id, filename=filename, path=str(dest), created_at=created_at)
    store.add(job, ttl_sec=settings.JOB_TTL_SEC)
    worker.submit(job_id, str(dest))
    log.info("job %s queued: %s (%.1f MB)", job_id, filename, size / 1e6)
    return JobCreated(job_id=job_id)


@app.get("/api/v1/jobs/{job_id}", response_model=JobResponse, response_model_exclude_none=True, dependencies=[Depends(require_api_key)])
async def get_job(job_id: str) -> JSONResponse:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return JSONResponse(job.to_response())


@app.get("/api/v1/rules", dependencies=[Depends(require_api_key)])
async def list_rules() -> dict:
    """Справочник типов нарушений и нормативных пунктов (для UI и документации)."""
    from pipeline.rules import DOCUMENTS, RULES

    return {"documents": DOCUMENTS, "rules": RULES}


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
