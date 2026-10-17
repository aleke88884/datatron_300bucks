"""Хранилище заданий в памяти (потокобезопасное). Одного процесса API достаточно по ТЗ:
истории проверок не требуется, результаты живут до перезапуска."""
from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Job:
    job_id: str
    filename: str
    path: str
    status: str = "processing"  # processing | done | failed
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    progress: dict = field(default_factory=lambda: {"stage": "queued", "percent": 0})
    result: Optional[dict] = None
    error: Optional[str] = None

    def to_response(self) -> dict[str, Any]:
        body: dict[str, Any] = {"job_id": self.job_id, "status": self.status}
        if self.status == "processing":
            body["progress"] = self.progress
        elif self.status == "done" and self.result is not None:
            body["verdict"] = self.result["verdict"]
            body["violations"] = self.result["violations"]
            meta = dict(self.result.get("meta", {}))
            if self.started_at and self.finished_at:
                meta["processing_time_sec"] = round(self.finished_at - self.started_at, 2)
            body["meta"] = meta
        elif self.status == "failed":
            body["error"] = self.error or "unknown error"
        return body


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def new_id(self) -> str:
        with self._lock:
            while True:
                jid = secrets.token_hex(4)  # как в примере ТЗ: "3f2a9c1e"
                if jid not in self._jobs:
                    return jid

    def add(self, job: Job) -> None:
        with self._lock:
            self._jobs[job.job_id] = job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, **fields: Any) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            for k, v in fields.items():
                setattr(job, k, v)

    def stats(self) -> dict[str, int]:
        with self._lock:
            out = {"processing": 0, "done": 0, "failed": 0}
            for j in self._jobs.values():
                out[j.status] = out.get(j.status, 0) + 1
            return out
