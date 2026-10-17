"""Очередь обработки: один фоновый воркер.

Задания выполняются строго по одному в отдельном дочернем процессе (multiprocessing, spawn).
Процесс живёт между заданиями (модели загружаются один раз в pipeline.warmup()), но если
задание не укладывается в JOB_TIMEOUT_SEC, процесс убивается, задание помечается failed,
а для следующего задания процесс поднимается заново. Поток в том же процессе убить нельзя,
поэтому именно процесс — единственный способ гарантировать лимит ТЗ (5 минут на видео).
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import os
import queue
import threading
import time
import traceback
from typing import Optional

from .config import settings
from .store import JobStore

log = logging.getLogger("worker")


def _child_main(task_q, result_q, log_level: str) -> None:  # pragma: no cover - отдельный процесс
    logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s [pipeline] %(message)s")
    try:
        from pipeline import analyze_video, warmup

        warmup()
        result_q.put(("ready", None, None))
    except Exception as exc:  # noqa: BLE001
        result_q.put(("fatal", None, {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}))
        return

    while True:
        item = task_q.get()
        if item is None:
            return
        job_id, path = item

        def progress(stage: str, percent: Optional[int] = None, _jid=job_id) -> None:
            result_q.put(("progress", _jid, {"stage": stage, "percent": percent}))

        try:
            result = analyze_video(path, progress=progress)
            result_q.put(("done", job_id, result))
        except Exception as exc:  # noqa: BLE001
            result_q.put(("failed", job_id, {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}))


class AnalysisWorker:
    def __init__(self, store: JobStore) -> None:
        self.store = store
        self._ctx = mp.get_context("spawn")
        self._pending: "queue.Queue[tuple[str, str] | None]" = queue.Queue()
        self._proc: Optional[mp.process.BaseProcess] = None
        self._task_q = None
        self._result_q = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    # --- публичный API -------------------------------------------------
    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="analysis-dispatcher", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._pending.put(None)
        self._kill_child()

    def submit(self, job_id: str, path: str) -> None:
        self._pending.put((job_id, path))

    @property
    def queue_size(self) -> int:
        return self._pending.qsize()

    @property
    def child_alive(self) -> bool:
        return bool(self._proc and self._proc.is_alive())

    # --- внутреннее ----------------------------------------------------
    def _ensure_child(self, deadline: float) -> bool:
        if self.child_alive:
            return True
        self._task_q = self._ctx.Queue()
        self._result_q = self._ctx.Queue()
        self._proc = self._ctx.Process(
            target=_child_main, args=(self._task_q, self._result_q, settings.LOG_LEVEL), name="analysis-worker", daemon=True
        )
        self._proc.start()
        log.info("worker process started pid=%s", self._proc.pid)
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                log.error("worker warmup timed out")
                self._kill_child()
                return False
            try:
                kind, _, payload = self._result_q.get(timeout=min(1.0, remaining))
            except queue.Empty:
                if not self._proc.is_alive():
                    log.error("worker process died during warmup")
                    return False
                continue
            if kind == "ready":
                return True
            if kind == "fatal":
                log.error("worker failed to start: %s\n%s", payload.get("error"), payload.get("traceback"))
                self._kill_child()
                return False

    def _kill_child(self) -> None:
        proc = self._proc
        if proc is None:
            return
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=5)
        self._proc = None
        for q in (self._task_q, self._result_q):
            try:
                if q is not None:
                    q.close()
            except Exception:  # noqa: BLE001
                pass
        self._task_q = self._result_q = None

    def _cleanup_upload(self, path: str) -> None:
        if settings.KEEP_UPLOADS:
            return
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        except OSError as exc:
            log.warning("cannot remove upload %s: %s", path, exc)

    def _loop(self) -> None:
        while not self._stop.is_set():
            item = self._pending.get()
            if item is None:
                break
            job_id, path = item
            self._run_one(job_id, path)
            self._cleanup_upload(path)

    def _run_one(self, job_id: str, path: str) -> None:
        started = time.time()
        deadline = started + settings.JOB_TIMEOUT_SEC
        self.store.update(job_id, started_at=started, progress={"stage": "starting", "percent": 0})

        if not self._ensure_child(deadline):
            self.store.update(job_id, status="failed", error="worker process failed to start", finished_at=time.time())
            return

        self._task_q.put((job_id, path))
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                log.error("job %s timed out after %ss, killing worker", job_id, settings.JOB_TIMEOUT_SEC)
                self._kill_child()
                self.store.update(
                    job_id,
                    status="failed",
                    error=f"processing timeout ({settings.JOB_TIMEOUT_SEC} s)",
                    finished_at=time.time(),
                )
                return
            try:
                kind, jid, payload = self._result_q.get(timeout=min(1.0, remaining))
            except queue.Empty:
                if not self.child_alive:
                    log.error("worker process died while processing job %s", job_id)
                    self._kill_child()
                    self.store.update(job_id, status="failed", error="worker process crashed", finished_at=time.time())
                    return
                continue
            if jid != job_id:
                continue
            if kind == "progress":
                self.store.update(job_id, progress=payload)
            elif kind == "done":
                self.store.update(job_id, status="done", result=payload, finished_at=time.time(), progress={"stage": "done", "percent": 100})
                log.info("job %s done in %.1fs: %s, %d violation(s)", job_id, time.time() - started, payload.get("verdict"), len(payload.get("violations", [])))
                return
            elif kind == "failed":
                log.error("job %s failed: %s\n%s", job_id, payload.get("error"), payload.get("traceback", ""))
                self.store.update(job_id, status="failed", error=payload.get("error", "pipeline error"), finished_at=time.time())
                return
