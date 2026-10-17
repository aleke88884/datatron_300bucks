"""Бюджет времени на видео: мягкий и жёсткий дедлайны + адаптивный шаг выборки кадров.

ТЗ: видео, не обработанное за 5 минут, считается необработанным (потерянные баллы). Поэтому пайплайн
сам следит за временем и деградирует вместо того, чтобы упасть по таймауту:

  мягкий дедлайн (soft_ts)  — к этому моменту восприятие и состояния должны быть готовы; дальше только правила,
                              кадры-доказательства (по 1 кадру, если опаздываем) и без VLM;
  жёсткий дедлайн (hard_ts) — воркер убивает процесс и отдаёт то, что уже есть (см. app/worker.py).

Деградация постепенная (StrideGovernor): после нескольких кадров измеряется стоимость одного шага; если прогноз
завершения выходит за дедлайн этапа, шаг выборки кадров увеличивается (3 к/с → 1.5 → 1 к/с). Треки переживают это:
TRACK_MAX_GAP_SEC = 1.2 с. В крайнем случае этап обрезается по времени, а правила работают по тому, что есть.

Бюджет задаётся через контекст (current()), поэтому сигнатуры run_perception/extract_states не менялись;
без бюджета (офлайн-оценка, скрипты) поведение прежнее: шаг не меняется, ничего не обрезается.
Только время (секунды), никаких привязок к камере.
"""
from __future__ import annotations

import contextlib
import contextvars
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger("pipeline.budget")

PERCEPTION_SHARE = 0.7      # доля оставшегося до мягкого дедлайна времени, отданная детектору
TRUNCATE_MARGIN_SEC = 4.0   # перед мягким дедлайном обрезаем восприятие, чтобы осталось время на состояния
STATES_GRACE_SEC = 12.0     # состояния могут выйти за мягкий дедлайн на это время (до жёсткого запас ≥ 60 с)
MAX_STRIDE = 3              # шаг выборки ≤ 3× от номинального (3 к/с → 1 к/с)
MIN_EVIDENCE_LEFT_SEC = 20  # меньше этого до жёсткого дедлайна — кадры-доказательства по одному на нарушение
MIN_VLM_LEFT_SEC = 60       # VLM запускается, только если до жёсткого дедлайна ещё есть этот запас


@dataclass
class Budget:
    soft_ts: float                       # time.time(), к которому восприятие+состояния должны закончиться
    hard_ts: float                       # time.time(), после которого процесс будет убит воркером
    started: float = field(default_factory=time.time)
    reasons: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    @property
    def degraded(self) -> bool:
        return bool(self.reasons)

    def note(self, reason: str) -> None:
        if reason not in self.reasons:
            self.reasons.append(reason)
            log.warning("degraded: %s", reason)

    def left_soft(self) -> float:
        return self.soft_ts - time.time()

    def left_hard(self) -> float:
        return self.hard_ts - time.time()

    def perception_deadline(self) -> float:
        now = time.time()
        return now + PERCEPTION_SHARE * max(0.0, self.soft_ts - now)

    def perception_truncate_ts(self) -> float:
        return max(time.time() + 1.0, self.soft_ts - TRUNCATE_MARGIN_SEC)

    def states_deadline(self) -> float:
        return self.soft_ts

    def states_truncate_ts(self) -> float:
        return self.soft_ts + STATES_GRACE_SEC


_current: contextvars.ContextVar[Optional[Budget]] = contextvars.ContextVar("pipeline_budget", default=None)


def current() -> Optional[Budget]:
    return _current.get()


@contextlib.contextmanager
def scope(budget: Optional[Budget]):
    token = _current.set(budget)
    try:
        yield budget
    finally:
        _current.reset(token)


def make_budget(soft_ts: Optional[float], hard_ts: Optional[float]) -> Optional[Budget]:
    """Бюджет из абсолютных меток времени; без них — None (адаптация выключена)."""
    if soft_ts is None and hard_ts is None:
        return None
    now = time.time()
    if soft_ts is None:
        soft_ts = now + 0.7 * max(1.0, hard_ts - now)
    if hard_ts is None:
        hard_ts = soft_ts + 65.0
    return Budget(soft_ts=float(soft_ts), hard_ts=float(max(hard_ts, soft_ts)))


class StrideGovernor:
    """Адаптивный шаг выборки: `take(n)` — обрабатывать ли n-й узел сетки; `update(done, total)` после обработки.

    deadline_ts=None — governor пассивен (шаг 1, никогда не останавливается).
    """

    def __init__(self, label: str, deadline_ts: Optional[float], truncate_ts: Optional[float] = None,
                 k_max: int = MAX_STRIDE, window: int = 4, budget: Optional[Budget] = None):
        self.label = label
        self.deadline_ts = deadline_ts
        self.truncate_ts = truncate_ts
        self.k_max = k_max
        self.window = window
        self.budget = budget
        self.k = 1
        self.stopped = False
        self.processed = 0
        self._mark_t: Optional[float] = None
        self._mark_n = 0
        self.history: list[tuple[int, int]] = []   # (индекс сетки, новый шаг)

    @property
    def active(self) -> bool:
        return self.deadline_ts is not None

    def take(self, n: int) -> bool:
        return n % self.k == 0

    def update(self, grid_done: int, grid_total: int) -> bool:
        """Вызывается после обработки очередного узла. Возвращает False, если этап пора обрезать."""
        self.processed += 1
        if not self.active:
            return True
        now = time.time()
        if self.truncate_ts is not None and now > self.truncate_ts:
            self.stopped = True
            if self.budget is not None:
                self.budget.note(f"{self.label}: обрезано по времени на узле {grid_done}/{grid_total}")
            return False
        if self._mark_t is None:
            # первый кадр (прогрев сессии/кэшей) в оценку скорости не берём
            self._mark_t, self._mark_n = now, self.processed
            return True
        cnt = self.processed - self._mark_n
        if cnt >= self.window:
            per = (now - self._mark_t) / cnt
            remaining = max(0.0, (grid_total - grid_done) / self.k)
            projected = remaining * per
            left = self.deadline_ts - now
            if projected > left and self.k < self.k_max:
                self.k += 1
                self.history.append((grid_done, self.k))
                if self.budget is not None:
                    self.budget.note(f"{self.label}: шаг выборки увеличен до ×{self.k} (прогноз {projected:.0f} с при остатке {left:.0f} с)")
                else:
                    log.warning("%s: stride -> x%d (projected %.0fs, left %.0fs)", self.label, self.k, projected, left)
            self._mark_t, self._mark_n = now, self.processed
        return True

    def summary(self) -> dict:
        return {"stride": self.k, "stopped": self.stopped, "changes": [list(h) for h in self.history]}
