"""Бюджет времени: адаптивный шаг выборки (pipeline/budget.py) и деградация живого пайплайна на реальном видео."""
from __future__ import annotations

import pytest

from pipeline import budget as B

from .conftest import NEGATIVE_VIDEO, video_path


class FakeClock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def _simulate(monkeypatch, per_sample_sec: float, deadline_left: float, total: int = 200, truncate_left=None):
    """Прогон governor-а: каждый обработанный узел стоит per_sample_sec секунд модельного времени."""
    clock = FakeClock()
    monkeypatch.setattr(B.time, "time", clock)
    gov = B.StrideGovernor("тест", clock.t + deadline_left, None if truncate_left is None else clock.t + truncate_left)
    n = 0
    done = 0
    while n < total:
        if gov.take(n):
            clock.t += per_sample_sec
            done += 1
            if not gov.update(n, total):
                break
        n += 1
    return gov, clock.t - 1000.0, done


def test_governor_is_passive_without_deadline(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(B.time, "time", clock)
    gov = B.StrideGovernor("тест", None)
    for n in range(100):
        assert gov.take(n)
        clock.t += 10
        assert gov.update(n, 100)
    assert gov.k == 1 and not gov.stopped


def test_governor_keeps_stride_when_fast_enough(monkeypatch):
    gov, spent, done = _simulate(monkeypatch, per_sample_sec=0.1, deadline_left=60)
    assert gov.k == 1 and done == 200


def test_governor_degrades_to_fit_deadline(monkeypatch):
    # 200 узлов по 1 с = 200 с при остатке 120 с: нужен шаг ×2 (≈ 100 с)
    gov, spent, done = _simulate(monkeypatch, per_sample_sec=1.0, deadline_left=120)
    assert gov.k == 2 and not gov.stopped
    assert spent <= 120 and done < 200


def test_governor_caps_stride_and_truncates(monkeypatch):
    # даже при шаге ×3 не успеваем: после max stride этап обрезается по truncate_ts
    gov, spent, done = _simulate(monkeypatch, per_sample_sec=2.0, deadline_left=30, truncate_left=40)
    assert gov.k == B.MAX_STRIDE and gov.stopped
    assert spent <= 40 + 2.0


def test_make_budget_without_deadlines_is_none():
    assert B.make_budget(None, None) is None
    b = B.make_budget(None, 1e12)
    assert b is not None and b.soft_ts < b.hard_ts


@pytest.mark.dataset
def test_live_pipeline_degrades_but_answers(monkeypatch):
    """Реальное видео + реальный детектор при заведомо недостижимом мягком дедлайне: ответ валиден, meta.degraded."""
    import time

    monkeypatch.delenv("FEATURES_CACHE", raising=False)
    from pipeline import analyze_video

    path = str(video_path(NEGATIVE_VIDEO))
    now = time.time()
    res = analyze_video(path, deadline_ts=now + 3, hard_deadline_ts=now + 120)
    assert res["verdict"] in ("violation", "no_violation")
    assert res["meta"]["degraded"] is True and res["meta"]["degrade_reasons"]
    for v in res["violations"]:
        assert 1 <= len(v["evidence_frames"]) <= 3
