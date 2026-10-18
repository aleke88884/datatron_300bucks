"""Очередь обработки: пул дочерних процессов с гарантией ответа в пределах лимита ТЗ (5 минут на видео).

Устройство
  * WORKERS процессов (multiprocessing, spawn; по умолчанию 1 со всеми ядрами, 2 при ≥ 12 ядрах), поднимаются сразу при старте сервиса
    и прогреваются (веса грузятся один раз, pipeline.warmup()). У каждого процесса свой поток-диспетчер в API-процессе.
  * Часы идут с момента POST (job.created_at), а не с момента взятия из очереди: мягкий дедлайн
    (SOFT_DEADLINE_SEC) передаётся в пайплайн, и он сам деградирует (реже кадры, без VLM, см. pipeline/budget.py);
    жёсткий (JOB_TIMEOUT_SEC) — процесс убивается.
  * Задание никогда не заканчивается «failed» из-за медленной машины или сбоя детектора:
      - жёсткий дедлайн  → отдаём последний частичный результат пайплайна (правила и кадры уже посчитаны),
                           а если его нет — безопасный ответ no_violation с meta.degraded=true;
      - падение процесса / необработанное исключение → один повтор в «лёгком» режиме (правила по яркости),
                           дальше — тот же безопасный ответ;
      - «failed» остаётся только для файла, который невозможно декодировать (VideoDecodeError), и для ситуации,
        когда процесс вообще не стартует: это определённый результат с понятной причиной.
  * Очередь — не FIFO: при пачке загрузок короткие записи берутся первыми (больше видео успевает пройти полный
    анализ за 5 минут), но задание, прождавшее дольше QUEUE_AGING_SEC, идёт раньше новых. Жёсткий дедлайн никогда
    не выходит за 5 минут ТЗ от POST (TZ_LIMIT_SEC - TZ_MARGIN_SEC).
  * Планирование очереди (plan_share): если полный анализ всех ждущих (и ещё загружающихся) видео не укладывается в
    5 минут, каждое задание получает пропорциональную долю времени — пайплайн увеличивает шаг выборки кадров и
    покрывает всё видео, вместо того чтобы первые шли полностью, а последние обрезались по времени.
  * Любая деградация видна в ответе: meta.degraded=true, meta.degraded_reason (кратко) и meta.degrade_reasons (все
    причины); meta.analysis_complete=false — анализ покрыл не всё видео или не выполнялся (вердикт по умолчанию).
  * Загруженное видео удаляется после обработки в любом исходе (finally), при старте сервиса остатки чистятся.
  * Ошибки в ответе API не содержат путей сервера (в журнал пишется полный текст).
"""
from __future__ import annotations

import inspect
import logging
import multiprocessing as mp
import os
import queue
import re
import threading
import time
import traceback
from typing import Optional

from .config import TZ_LIMIT_SEC, TZ_MARGIN_SEC, settings
from .store import JobStore

log = logging.getLogger("worker")

WARMUP_TIMEOUT_SEC = 120.0
LIGHT_RETRY_MIN_LEFT_SEC = 12.0   # повтор в лёгком режиме, только если до жёсткого дедлайна осталось столько

_PATH_RE = re.compile(r"""(?:[A-Za-z]:)?(?:[\\/][^\s'":,;()\[\]]+){2,}""")
# признаки того, что анализ покрыл не всё видео (pipeline/budget.py: «обрезано по времени …»)
_INCOMPLETE_MARKERS = ("обрезано", "вердикт по умолчанию", "light mode", "лёгк")


def public_error(text: str) -> str:
    """Текст ошибки для ответа API: без абсолютных путей сервера и без многострочных трассировок."""
    first = str(text or "").strip().splitlines()[0] if str(text or "").strip() else "unknown error"
    return _PATH_RE.sub("<file>", first)[:300]


class _PendingQueue:
    """Очередь заданий с приоритетом: сначала прождавшие дольше aging_sec (по размеру), затем остальные по размеру.

    Для одновременной пачки это «кратчайшее задание первым»: при общем дедлайне от POST оно максимизирует число видео,
    успевших пройти полный анализ. Размер файла — оценка длительности записи (камеры одного типа, близкий битрейт).
    """

    def __init__(self, aging_sec: float) -> None:
        self._items: list[tuple[float, int, int, str, str]] = []   # (created, size, seq, job_id, path)
        self._cv = threading.Condition()
        self._seq = 0
        self._closed = False
        self.aging_sec = aging_sec

    def put(self, job_id: str, path: str, created: float, size: int) -> None:
        with self._cv:
            self._seq += 1
            self._items.append((created, size, self._seq, job_id, path))
            self._cv.notify()

    def close(self) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify_all()

    def get(self, timeout: float) -> Optional[tuple[str, str]]:
        """(job_id, path); None — очередь закрыта; queue.Empty — истёк timeout."""
        with self._cv:
            end = time.time() + timeout
            while not self._items and not self._closed:
                left = end - time.time()
                if left <= 0:
                    raise queue.Empty
                self._cv.wait(left)
            if self._closed:
                return None
            now = time.time()
            best = min(range(len(self._items)), key=lambda i: self._key(self._items[i], now))
            _, _, _, job_id, path = self._items.pop(best)
            return job_id, path

    def _key(self, item: tuple, now: float) -> tuple:
        created, size, seq, _, _ = item
        aged = now - created >= self.aging_sec
        return (0 if aged else 1, size, seq)

    def qsize(self) -> int:
        with self._cv:
            return len(self._items)

    def snapshot(self) -> list[tuple[float, int]]:
        """(created, size) ожидающих заданий в порядке, в котором они будут взяты."""
        with self._cv:
            now = time.time()
            return [(c, sz) for c, sz, *_ in sorted(self._items, key=lambda it: self._key(it, now))]


class CostModel:
    """Оценка времени полного анализа по размеру файла: OVERHEAD + sec_per_mb * МБ.

    Коэффициент уточняется по завершённым НЕдеградированным заданиям (скользящее среднее), так что модель
    подстраивается под машину. Стартовое значение — замер на M3 8 ГБ, 1 воркер × 8 потоков: Х-14 (27.4 МБ, 143 с)
    — 165 с под фоновой нагрузкой."""

    OVERHEAD_SEC = 4.0

    def __init__(self, sec_per_mb: float = 6.0) -> None:
        self.sec_per_mb = sec_per_mb
        self._lock = threading.Lock()

    def estimate(self, size_bytes: int) -> float:
        return self.OVERHEAD_SEC + self.sec_per_mb * max(0, size_bytes) / 1e6

    def observe(self, size_bytes: int, seconds: float) -> None:
        mb = size_bytes / 1e6
        if mb < 2.0 or seconds <= self.OVERHEAD_SEC:   # слишком короткие записи — шумная оценка
            return
        rate = (seconds - self.OVERHEAD_SEC) / mb
        with self._lock:
            self.sec_per_mb = 0.6 * self.sec_per_mb + 0.4 * rate


# Мягкий дедлайн при сжатии = SOFT_FACTOR × (доля × полное время). pipeline/budget.py отдаёт детектору 70 % времени
# до мягкого дедлайна, а детекция — ~85 % полного анализа (замер Х-14: 140 из 165 с): 0.85 / 0.7 ≈ 1.2.
SOFT_FACTOR = 1.2
MIN_SHARE = 0.34    # меньше ~1/3 полного времени шаг выборки уже не спасает (×3 максимум) — дальше решает обрезка


def _child_main(task_q, result_q, log_level: str, threads: int = 0) -> None:  # pragma: no cover - отдельный процесс
    # Потоки задаём ДО импорта numpy/onnxruntime/cv2: при нескольких процессах иначе каждый берёт все ядра.
    if threads and threads > 0:
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "ORT_INTRA_THREADS"):
            os.environ[var] = str(threads)
    logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s [pipeline] %(message)s")
    try:
        from pipeline import analyze_video, warmup

        try:
            import cv2

            cv2.setNumThreads(max(1, min(2, threads or 2)))
        except Exception:  # noqa: BLE001
            pass
        warmup()
        params = inspect.signature(analyze_video).parameters
        takes_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
        result_q.put(("ready", None, None))
    except Exception as exc:  # noqa: BLE001
        result_q.put(("fatal", None, {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}))
        return

    def supports(name: str) -> bool:
        return takes_kw or name in params

    while True:
        item = task_q.get()
        if item is None:
            return
        job_id, path, opts = item

        def progress(stage: str, percent: Optional[int] = None, _jid=job_id) -> None:
            result_q.put(("progress", _jid, {"stage": stage, "percent": percent}))

        def partial(result: dict, _jid=job_id) -> None:
            result_q.put(("partial", _jid, result))

        kwargs: dict = {}
        for src, dst in (("soft_ts", "deadline_ts"), ("hard_ts", "hard_deadline_ts")):
            if opts.get(src) is not None and supports(dst):
                kwargs[dst] = opts[src]
        if supports("partial"):
            kwargs["partial"] = partial
        if opts.get("light") and supports("light"):
            kwargs["light"] = True

        try:
            result = analyze_video(path, progress=progress, **kwargs)
            result_q.put(("done", job_id, result))
        except Exception as exc:  # noqa: BLE001
            result_q.put(("failed", job_id, {
                "error": f"{type(exc).__name__}: {exc}", "kind": type(exc).__name__, "traceback": traceback.format_exc(),
            }))
        finally:
            try:
                from pipeline.video import release_temp_copies

                release_temp_copies()
            except Exception:  # noqa: BLE001
                pass


class _Slot:
    """Один дочерний процесс и его очереди. Принадлежит ровно одному потоку-диспетчеру."""

    def __init__(self, idx: int) -> None:
        self.idx = idx
        self.proc: Optional[mp.process.BaseProcess] = None
        self.task_q = None
        self.result_q = None
        self.ready = False          # процесс прогрет (получено "ready")

    @property
    def alive(self) -> bool:
        return bool(self.proc and self.proc.is_alive())


class AnalysisWorker:
    def __init__(self, store: JobStore, workers: Optional[int] = None) -> None:
        self.store = store
        self.n_workers = workers or settings.worker_count
        self._ctx = mp.get_context("spawn")
        self._pending = _PendingQueue(settings.QUEUE_AGING_SEC)
        self.cost = CostModel()
        self._sizes: dict[str, int] = {}
        self._plan_notes: dict[str, str] = {}
        self._inflight: dict[int, tuple[float, int]] = {}   # загрузки, ещё идущие в API: token -> (начало, объявленный размер)
        self._inflight_lock = threading.Lock()
        self._inflight_seq = 0
        self._slots = [_Slot(i) for i in range(self.n_workers)]
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()

    def upload_started(self, declared_size: int) -> int:
        """API начал принимать файл: учитываем его в планировании ещё до постановки в очередь."""
        with self._inflight_lock:
            self._inflight_seq += 1
            self._inflight[self._inflight_seq] = (time.time(), max(0, declared_size))
            return self._inflight_seq

    def upload_finished(self, token: int) -> None:
        with self._inflight_lock:
            self._inflight.pop(token, None)

    # --- публичный API -------------------------------------------------
    def start(self) -> None:
        for slot in self._slots:
            t = threading.Thread(target=self._slot_loop, args=(slot,), name=f"analysis-dispatcher-{slot.idx}", daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self) -> None:
        self._stop.set()
        self._pending.close()
        for slot in self._slots:
            self._kill_child(slot)

    def submit(self, job_id: str, path: str, created: Optional[float] = None, size: Optional[int] = None) -> None:
        if size is None:
            try:
                size = os.path.getsize(path)
            except OSError:
                size = 0
        self._sizes[job_id] = size
        self._pending.put(job_id, path, created if created is not None else time.time(), size)

    @property
    def queue_size(self) -> int:
        return self._pending.qsize()

    @property
    def child_alive(self) -> bool:
        return any(s.alive for s in self._slots)

    @property
    def workers_alive(self) -> int:
        return sum(1 for s in self._slots if s.alive and s.ready)

    # --- процессы ------------------------------------------------------
    def _ensure_child(self, slot: _Slot, deadline: float) -> bool:
        if slot.alive:
            return True
        self._kill_child(slot)
        slot.task_q = self._ctx.Queue()
        slot.result_q = self._ctx.Queue()
        slot.proc = self._ctx.Process(
            target=_child_main,
            args=(slot.task_q, slot.result_q, settings.LOG_LEVEL, settings.threads_per_worker),
            name=f"analysis-worker-{slot.idx}",
            daemon=True,
        )
        slot.proc.start()
        log.info("worker process #%d started pid=%s (threads=%d)", slot.idx, slot.proc.pid, settings.threads_per_worker)
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                log.error("worker #%d warmup timed out", slot.idx)
                self._kill_child(slot)
                return False
            try:
                kind, _, payload = slot.result_q.get(timeout=min(1.0, remaining))
            except queue.Empty:
                if not slot.proc.is_alive():
                    log.error("worker process #%d died during warmup", slot.idx)
                    self._kill_child(slot)
                    return False
                continue
            if kind == "ready":
                slot.ready = True
                return True
            if kind == "fatal":
                log.error("worker #%d failed to start: %s\n%s", slot.idx, payload.get("error"), payload.get("traceback"))
                self._kill_child(slot)
                return False

    def _kill_child(self, slot: _Slot) -> None:
        proc = slot.proc
        if proc is not None:
            try:
                if proc.is_alive():
                    proc.kill()
                proc.join(timeout=5)
            except Exception:  # noqa: BLE001
                pass
        slot.proc = None
        slot.ready = False
        for q in (slot.task_q, slot.result_q):
            try:
                if q is not None:
                    q.close()
                    q.cancel_join_thread()
            except Exception:  # noqa: BLE001
                pass
        slot.task_q = slot.result_q = None

    # --- диспетчер -----------------------------------------------------
    def _cleanup_upload(self, path: str) -> None:
        if settings.KEEP_UPLOADS:
            return
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        except OSError as exc:
            log.warning("cannot remove upload %s: %s", path, exc)

    def _slot_loop(self, slot: _Slot) -> None:
        # прогрев сразу при старте: загрузка весов не должна съедать время первого задания
        try:
            self._ensure_child(slot, time.time() + WARMUP_TIMEOUT_SEC)
        except Exception:  # noqa: BLE001
            log.exception("eager warmup of worker #%d failed", slot.idx)
        while not self._stop.is_set():
            try:
                item = self._pending.get(timeout=1.0)
            except queue.Empty:
                self._revive(slot)
                continue
            if item is None:
                break
            job_id, path = item
            try:
                self._run_one(slot, job_id, path)
            except Exception as exc:  # noqa: BLE001 — диспетчер не должен умереть ни при каких обстоятельствах
                log.exception("dispatcher: job %s crashed the dispatcher step: %s", job_id, exc)
                self._kill_child(slot)
            finally:
                self._cleanup_upload(path)
                self._ensure_terminal(job_id, "внутренняя ошибка диспетчера")
            self._revive(slot)

    def _revive(self, slot: _Slot) -> None:
        """Подготовить процесс к следующему заданию (после убийства по таймауту/падения)."""
        if self._stop.is_set() or slot.alive:
            return
        try:
            self._ensure_child(slot, time.time() + WARMUP_TIMEOUT_SEC)
        except Exception:  # noqa: BLE001
            log.exception("cannot revive worker #%d", slot.idx)

    # --- завершение задания --------------------------------------------
    def _finish_done(self, job_id: str, result: dict, started: float, note: Optional[str] = None) -> None:
        meta = result.setdefault("meta", {})
        if not isinstance(meta, dict):
            meta = result["meta"] = {}
        reasons = [public_error(r) for r in (meta.get("degrade_reasons") or [])]
        plan_note = self._plan_notes.pop(job_id, None)
        if plan_note and (reasons or note):   # сжатие по очереди упоминаем, только если пайплайн и правда деградировал
            reasons.insert(0, plan_note)
        if note:
            reasons.append(public_error(note))
        degraded = bool(meta.get("degraded")) or bool(reasons)
        meta["degraded"] = degraded
        meta["degrade_reasons"] = reasons
        if degraded:
            meta["degraded_reason"] = "; ".join(reasons) if reasons else "анализ выполнен в упрощённом режиме"
            incomplete = bool(meta.get("fallback")) or bool(meta.get("light")) or any(
                m in r for r in reasons for m in _INCOMPLETE_MARKERS)
            meta["analysis_complete"] = not incomplete
        else:
            meta.pop("degraded_reason", None)
            meta["analysis_complete"] = True
        job = self.store.get(job_id)
        if job is not None and job.started_at:
            meta["queue_wait_sec"] = round(max(0.0, job.started_at - job.created_at), 2)
        size = self._sizes.pop(job_id, None)
        if size and not degraded and isinstance(meta.get("pipeline_time_sec"), (int, float)):
            self.cost.observe(size, float(meta["pipeline_time_sec"]))
        self.store.update(job_id, status="done", result=result, finished_at=time.time(), progress={"stage": "done", "percent": 100})
        log.info("job %s done in %.1fs: %s, %d violation(s)%s", job_id, time.time() - started, result.get("verdict"),
                 len(result.get("violations", [])), f" [degraded: {note}]" if note else "")

    def _fallback_result(self, reason: str) -> dict:
        # Вердикт по умолчанию — только когда анализ не дал вообще никакого результата; это всегда помечено:
        # meta.degraded=true, meta.fallback=true, meta.analysis_complete=false.
        return {"verdict": "no_violation", "violations": [], "meta": {
            "degraded": True, "fallback": True, "analysis_complete": False, "degrade_reasons": [reason]}}

    def _ensure_terminal(self, job_id: str, reason: str) -> None:
        job = self.store.get(job_id)
        if job is not None and job.status == "processing":
            log.error("job %s left without a terminal status, answering with fallback: %s", job_id, reason)
            self._finish_done(job_id, self._fallback_result(f"{reason}: вердикт по умолчанию"), time.time())

    def plan_share(self, job_id: str, created: float, now: float) -> tuple[float, float]:
        """Доля полного анализа, которую можно дать этому заданию, чтобы ждущие в очереди тоже успели (≤ 1), и оценка
        полного времени этого задания.

        Для каждого префикса «это задание + k следующих в очереди» проверяем, укладывается ли их суммарное полное время
        (с учётом числа воркеров) в дедлайн k-го. Если нет — все сжимаются пропорционально (пайплайн увеличит шаг
        выборки кадров, покрывая всё видео), вместо того чтобы первые шли полностью, а последние обрезались.
        """
        timeout = float(settings.JOB_TIMEOUT_SEC)
        own = self.cost.estimate(self._sizes.get(job_id, 0))
        with self._inflight_lock:
            inflight = list(self._inflight.values())
        jobs = ([(created, own)] + [(c, self.cost.estimate(sz)) for c, sz in self._pending.snapshot()]
                + [(c, self.cost.estimate(sz)) for c, sz in sorted(inflight, key=lambda x: x[1])])
        share, cum = 1.0, 0.0
        for c, cost in jobs:
            cum += cost
            window = (c + timeout - now) * self.n_workers * 0.95
            if window <= 0:
                continue   # это задание уже не успевает — оно не должно сжимать остальных
            share = min(share, window / cum)
        return max(MIN_SHARE, min(1.0, share)), own

    def _run_one(self, slot: _Slot, job_id: str, path: str) -> None:
        job = self.store.get(job_id)
        created = job.created_at if job is not None else time.time()
        started = time.time()
        timeout = float(settings.JOB_TIMEOUT_SEC)
        min_work = settings.MIN_WORK_SEC if settings.MIN_WORK_SEC > 0 else 0.1 * timeout
        # Часы идут с момента POST; но заданию, долго простоявшему в очереди, оставляем минимум рабочего времени.
        soft = max(created + settings.SOFT_DEADLINE_SEC, started + min_work)
        share, est = 1.0, 0.0
        try:
            share, est = self.plan_share(job_id, created, started)
        except Exception:  # noqa: BLE001 — планировщик не должен ломать задание
            log.exception("queue planner failed for job %s", job_id)
        if share < 1.0:
            planned = started + max(min_work, SOFT_FACTOR * share * est)
            if planned < soft:
                log.info("job %s: queue of %d waiting, giving %.0f%% of full analysis (est %.0fs): soft deadline in %.0fs instead of %.0fs",
                         job_id, self._pending.qsize(), 100 * share, est, planned - started, soft - started)
                soft = planned
        hard = max(created + timeout, soft + min_work)
        if share < 1.0:
            self._plan_notes[job_id] = (f"очередь из нескольких видео: анализ ускорен (≈{round(100 * share)}% полного времени), "
                                        "чтобы все задания уложились в 5 минут")
        # дальше 5 минут ТЗ от POST ответ уже не засчитывается: не тянем дедлайн за этот предел
        tz_cap = created + TZ_LIMIT_SEC - TZ_MARGIN_SEC
        if hard > tz_cap:
            hard = max(started + 5.0, tz_cap)
            soft = min(soft, started + 0.7 * max(1.0, hard - started))
        self.store.update(job_id, started_at=started, progress={"stage": "starting", "percent": 0})
        if started - created > 2:
            log.info("job %s waited %.1fs in queue; soft deadline in %.0fs, hard in %.0fs", job_id, started - created, soft - started, hard - started)

        if not self._ensure_child(slot, hard):
            self.store.update(job_id, status="failed", error="analysis worker failed to start", finished_at=time.time())
            return

        slot.task_q.put((job_id, path, {"soft_ts": soft, "hard_ts": hard, "light": False}))
        partial: Optional[dict] = None
        light = False
        while True:
            remaining = hard - time.time()
            if remaining <= 0:
                log.error("job %s hit the hard deadline (%.0fs after POST), killing worker #%d", job_id, time.time() - created, slot.idx)
                self._kill_child(slot)
                if partial is not None:
                    self._finish_done(job_id, partial, started, "жёсткий дедлайн: отдан результат, готовый к этому моменту")
                else:
                    self._finish_done(job_id, self._fallback_result("жёсткий дедлайн: анализ не завершён, вердикт по умолчанию"), started)
                return
            try:
                kind, jid, payload = slot.result_q.get(timeout=min(1.0, remaining))
            except queue.Empty:
                if not slot.alive:
                    log.error("worker process #%d died while processing job %s", slot.idx, job_id)
                    self._kill_child(slot)
                    if partial is None and self._retry_light(slot, job_id, path, soft, hard, light):
                        light = True
                        continue
                    self._finish_unrecoverable(job_id, partial, started, "процесс анализа аварийно завершился")
                    return
                continue
            except (OSError, ValueError, EOFError):  # очередь закрыта/повреждена
                self._kill_child(slot)
                self._finish_unrecoverable(job_id, partial, started, "сбой канала связи с процессом анализа")
                return
            if jid != job_id:
                continue
            if kind == "progress":
                self.store.update(job_id, progress=payload)
            elif kind == "partial":
                partial = payload
            elif kind == "done":
                self._finish_done(job_id, payload, started)
                return
            elif kind == "failed":
                err = payload.get("error", "pipeline error")
                log.error("job %s failed: %s\n%s", job_id, err, payload.get("traceback", ""))
                if payload.get("kind") == "VideoDecodeError":
                    # файл невозможно декодировать: это определённый результат, а не сбой сервиса.
                    # Путь к файлу на сервере в ответ не попадает (он есть в журнале).
                    self.store.update(job_id, status="failed", error=public_error(err), finished_at=time.time())
                    return
                if partial is None and not light and self._retry_light(slot, job_id, path, soft, hard, light, same_child=True):
                    light = True
                    continue
                self._finish_unrecoverable(job_id, partial, started, f"ошибка анализа ({err})")
                return

    def _retry_light(self, slot: _Slot, job_id: str, path: str, soft: float, hard: float, already_light: bool, same_child: bool = False) -> bool:
        """Повтор в лёгком режиме (правила по яркости), если ещё есть время. True — задание отправлено."""
        if already_light or hard - time.time() < LIGHT_RETRY_MIN_LEFT_SEC or not os.path.exists(path):
            return False
        try:
            if not same_child and not self._ensure_child(slot, hard - 5):
                return False
            if not slot.alive:
                return False
            log.warning("job %s: retrying in light mode on worker #%d", job_id, slot.idx)
            slot.task_q.put((job_id, path, {"soft_ts": soft, "hard_ts": hard, "light": True}))
            return True
        except Exception:  # noqa: BLE001
            log.exception("light retry failed to start")
            return False

    def _finish_unrecoverable(self, job_id: str, partial: Optional[dict], started: float, reason: str) -> None:
        if partial is not None:
            self._finish_done(job_id, partial, started, reason)
        else:
            self._finish_done(job_id, self._fallback_result(f"{reason}: вердикт по умолчанию"), started)
