"""Уточнение таймкодов (pipeline.timing) и точное чтение кадров-доказательств (video.read_frames_at).

Синтетические траектории — без датасета и детектора; проверка на датасете — в конце (пропускается без видео/кеша).
"""
from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from .conftest import SERVICE_DIR, VIDEOS_DIR

if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

from pipeline import timing  # noqa: E402
from pipeline.perception import PersonObs, Track  # noqa: E402

FPS = 3.0


def _track(tid: int, path: list[tuple[float, float, float, float]], inside: bool = True) -> Track:
    """path: [(t, foot_x, foot_y, h)] → трек с рамками шириной 0.4 h."""
    tr = Track(id=tid)
    for t, fx, fy, h in path:
        tr.obs.append(PersonObs(t=t, xyxy=(fx - 0.2 * h, fy - h, fx + 0.2 * h, fy), score=0.9, inside=inside, track_id=tid, state=1))
    return tr


def _seg(t0: float, t1: float, p0: tuple[float, float], p1: tuple[float, float], h0: float, h1: float | None = None):
    h1 = h0 if h1 is None else h1
    n = max(1, int(round((t1 - t0) * FPS)))
    out = []
    for k in range(n):
        a = k / n
        out.append((t0 + k / FPS, p0[0] + a * (p1[0] - p0[0]), p0[1] + a * (p1[1] - p0[1]), h0 + a * (h1 - h0)))
    return out


def _leave_path():
    """Сидит за столом 0–5 с, встаёт, подходит к сейфу (6–8.5 с стоит), идёт к двери, 1 с у двери, уходит."""
    desk, safe, door = (300.0, 400.0), (260.0, 390.0), (120.0, 360.0)
    p = []
    p += _seg(0.0, 5.0, desk, desk, 110)                    # сидит
    p += _seg(5.0, 6.0, desk, safe, 110, 190)               # встаёт и подходит к сейфу
    p += _seg(6.0, 8.5, safe, safe, 190)                    # работает у сейфа 2.5 с
    p += _seg(8.5, 10.0, safe, door, 190)                   # идёт к двери
    p += _seg(10.0, 11.0, door, door, 190)                  # открывает дверь (пауза 1 с)
    p += _seg(11.0, 11.7, door, (60.0, 350.0), 190)         # выходит
    return p


def test_stops_split_sitting_work_and_walk():
    tr = _track(1, _leave_path())
    st = [s for s in timing.stops(tr.obs) if s.dur >= 1.0]
    assert [round(s.t0) for s in st][:2] == [0, 6]
    assert st[1].dur >= 2.0


def test_refine_leave_onset_is_end_of_last_work_stop():
    """Таймкод ухода — кассир трогается от сейфа (≈8.7 с), а не встаёт (5 с) и не пропадает (11.7 с);
    короткая пауза у двери (1 с) за работу не считается."""
    tr = _track(1, _leave_path())
    sc = SimpleNamespace(cashier_track_ids=[1])
    t = timing.refine_leave_onset(sc, [tr], t_gone=12.0)
    assert t is not None and 8.4 <= t <= 9.0, t


def test_refine_leave_onset_ignores_long_pause_far_from_work_place():
    """Остановка вдали от мест, где кассир работал (у двери, ≥ 2 с), — не «рабочая»: берём остановку у сейфа."""
    p = _leave_path()
    p = [x for x in p if x[0] < 10.0] + _seg(10.0, 12.0, (120.0, 360.0), (120.0, 360.0), 190) + _seg(12.0, 12.7, (120.0, 360.0), (60.0, 350.0), 190)
    sc = SimpleNamespace(cashier_track_ids=[1])
    t = timing.refine_leave_onset(sc, [_track(1, p)], t_gone=13.0)
    assert t is not None and 8.4 <= t <= 9.0, t


def test_refine_leave_onset_none_without_tracks():
    sc = SimpleNamespace(cashier_track_ids=[1])
    assert timing.refine_leave_onset(sc, None, 10.0) is None
    assert timing.refine_leave_onset(SimpleNamespace(cashier_track_ids=[]), [_track(1, _leave_path())], 12.0) is None


def test_refine_entry_time_reaches_destination():
    """Вошедший появляется в проёме (16 с), 2 с идёт к столу и останавливается: таймкод — когда дошёл."""
    p = _seg(16.0, 17.0, (170.0, 310.0), (165.0, 320.0), 180) + _seg(17.0, 19.0, (165.0, 320.0), (50.0, 420.0), 180, 230) \
        + _seg(19.0, 23.0, (50.0, 420.0), (52.0, 422.0), 230)
    newcomer = _track(6, p)
    client = _track(3, _seg(10.0, 23.0, (190.0, 367.0), (190.0, 367.0), 120))
    sc = SimpleNamespace(visitor_track_ids=[3, 6])
    res = timing.refine_entry_time(sc, [client, newcomer], t_start=16.0, t_end=23.0)
    assert res is not None and res[1] == 6
    assert 18.3 <= res[0] <= 19.2, res


def test_visitors_at_desk_frac_distinguishes_client_at_desk_from_person_behind():
    desk = {"bbox": [50, 366, 300, 500]}
    sc = SimpleNamespace(anchors={"desk": desk}, visitor_track_ids=[2])
    at_desk = _track(2, _seg(12.0, 16.0, (75.0, 390.0), (75.0, 390.0), 210))
    behind = _track(2, _seg(12.0, 16.0, (160.0, 316.0), (160.0, 316.0), 150))
    assert timing.visitors_at_desk_frac(sc, [at_desk], 12.0, 16.0)[0] == pytest.approx(1.0)
    assert timing.visitors_at_desk_frac(sc, [behind], 12.0, 16.0)[0] == pytest.approx(0.0)


def test_refine_object_onset_requires_consecutive_detections(monkeypatch):
    info = SimpleNamespace(width=464, height=848, fps=30.0)
    dets = [(10.0, [("cup", 0.55, [100, 300, 120, 330])]), (10.2, []), (10.4, [("cup", 0.2, [100, 300, 120, 330])]),
            (10.6, [("cup", 0.4, [100, 300, 120, 330])]), (10.8, [("cup", 0.8, [100, 300, 120, 330])]),
            (11.0, [("cup", 0.7, [101, 300, 121, 330])]), (11.2, [("cup", 0.9, [101, 300, 121, 330])])]
    monkeypatch.delenv("FEATURES_CACHE", raising=False)
    monkeypatch.setattr(timing, "_dense_detections", lambda *a, **k: dets)
    res = timing.refine_object_onset("x.mov", info, 13.0, (20, 250, 320, 520), {"cup"})
    assert res is not None and res[0] == pytest.approx(10.6)
    assert res[1] == (100, 300, 120, 330)


def test_read_frames_at_is_exact(tmp_path):
    """Кадры по запрошенным моментам совпадают с последовательным декодированием (номер кадра зашит в яркость)."""
    import cv2

    from pipeline.video import iter_frames, probe, read_frame_at, read_frames_at

    path = tmp_path / "idx.avi"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (64, 48))
    for k in range(90):
        w.write(np.full((48, 64, 3), (k * 2) % 256, np.uint8))
    w.release()
    info = probe(str(path))
    times = [2.5, 0.0, 1.0, 1.0, 10.0]   # повтор, произвольный порядок, момент за концом записи
    frames = read_frames_at(str(path), times, info)
    seq = {idx: fr for idx, _t, fr in iter_frames(str(path))}
    assert len(frames) == len(times)
    for t, fr in zip(times[:4], frames[:4]):
        k = int(round(t * info.fps))
        assert fr is not None and np.abs(fr.astype(int) - seq[k].astype(int)).mean() < 2
    assert frames[4] is not None and np.abs(frames[4].astype(int) - seq[max(seq)].astype(int)).mean() < 2
    single = read_frame_at(str(path), 1.0, info)
    assert np.abs(single.astype(int) - seq[30].astype(int)).mean() < 2


@pytest.mark.dataset
def test_read_frames_at_matches_sequential_on_variable_fps_video():
    """Х-11 (fps 29.88): позиционирование по индексу кадра попадало на ~7 с позже; read_frames_at — точно."""
    from pipeline.video import iter_frames, probe, read_frames_at

    path = VIDEOS_DIR / "Х-11.mov"
    if not path.exists():
        pytest.skip("нет Х-11.mov")
    info = probe(str(path))
    times = [3.0, 8.0, 13.0]
    want = {int(round(t * info.fps)) for t in times}
    seq = {}
    for idx, _t, fr in iter_frames(str(path)):
        if idx in want:
            seq[idx] = fr
        if idx > max(want):
            break
    for t, fr in zip(times, read_frames_at(str(path), times, info)):
        assert np.abs(fr.astype(int) - seq[int(round(t * info.fps))].astype(int)).mean() < 1.0
