"""T13 «дверь кассы оставлена открытой»: правило на синтетических рядах + живой путь pipeline/door.py на видео датасета."""
from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from .conftest import RESULTS_DIR, SERVICE_DIR, VIDEOS_DIR

if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

from pipeline.context import VideoContext  # noqa: E402
from pipeline.rules import RULES  # noqa: E402
from pipeline.rules_door import OPEN_HOLD_SEC, rule_t13_door  # noqa: E402
from pipeline.video import VideoInfo  # noqa: E402

FPS = 4.0


def _ctx(door: dict | None, duration: float = 20.0) -> VideoContext:
    info = VideoInfo(path="x", width=464, height=848, fps=30.0, frame_count=int(duration * 30), duration_sec=duration)
    ctx = VideoContext(path="x", info=info, bright_t=np.linspace(0, duration, 50), bright_v=np.full(50, 130.0))
    if door is not None:
        ctx.cache["door"] = door
    return ctx


def _door(open_from: float, close_at: float | None, duration: float = 20.0, crossings=None, bright=None, peak=0.6) -> dict:
    """Синтетический ряд: дверь закрыта (0.0) → открыта (peak) с open_from → закрыта с close_at (или до конца)."""
    t = np.arange(0.0, duration, 1.0 / FPS)
    ch = np.where((t >= open_from) & ((t < close_at) if close_at is not None else True), peak, 0.01)
    return {
        "version": 1, "fps": FPS, "scale": 0.5, "found": True, "roi": [0.0, 28.0, 128.0, 360.0], "roi_frac": 0.1,
        "members": [], "t_first_motion": open_from - 0.5, "ref_window": [0.0, open_from - 0.75], "peak": peak,
        "t": [round(float(v), 2) for v in t], "change": [round(float(v), 3) for v in ch],
        "bright_ratio": bright if bright is not None else [1.0] * len(t),
        "crossings": crossings if crossings is not None else [{"kind": "in", "t": 2.0, "track": 1, "t_cross": open_from + 1.5, "cover": 0.2}],
    }


def test_door_left_open_after_entry_fires_at_crossing():
    ev = rule_t13_door(_ctx(_door(4.0, None)))
    assert len(ev) == 1
    e = ev[0]
    assert e.type_code == "T13" and e.timecode_sec == pytest.approx(5.5)
    assert e.confidence >= 0.5 and e.end_sec >= 19.0
    assert "не закрывает" in e.description and len(e.evidence_times) == 3
    assert all(b == (0, 28, 128, 360) for b in e.extras["evidence_boxes"])


def test_door_closed_soon_after_entry_is_clean():
    """Как в чистом Х-31: открыли, вошли, через 3 с закрыли."""
    assert rule_t13_door(_ctx(_door(4.0, 7.0))) == []


def test_hold_threshold_boundary():
    assert rule_t13_door(_ctx(_door(4.0, 4.0 + OPEN_HOLD_SEC - 1.0))) == []
    assert len(rule_t13_door(_ctx(_door(4.0, 4.0 + OPEN_HOLD_SEC + 1.0)))) == 1


def test_ajar_door_below_open_ratio_is_not_open():
    """Неплотно прикрытая дверь (Х-39): изменение ~0.5 пика первого открывания — не «открыта»."""
    d = _door(4.0, 7.0)
    d["change"] = [0.6 if 4.0 <= t < 7.0 else (0.33 if t >= 7.0 else 0.01) for t in d["t"]]
    assert rule_t13_door(_ctx(d)) == []


def test_lights_off_frames_are_not_judged():
    """Свет выключен: яркость кадра падает относительно эталона — ряд двери не судим (Х-11/Х-12)."""
    d = _door(4.0, None)
    d["bright_ratio"] = [1.0 if t < 4.5 else 0.6 for t in d["t"]]
    assert rule_t13_door(_ctx(d)) == []


def test_polarity_guard_when_video_starts_with_open_door():
    """Вход через проём раньше первого движения створки → эталон был открытой дверью (её потом закрыли) — молчим."""
    d = _door(6.0, None, crossings=[{"kind": "in", "t": 1.0, "track": 1, "t_cross": 2.0, "cover": 0.3}])
    assert rule_t13_door(_ctx(d)) == []


def test_masked_run_needs_visibility():
    """Дверь почти всё время закрыта людьми (None) — не судим."""
    d = _door(4.0, None)
    d["change"] = [c if (i % 3 == 0 or t < 4.0) else None for i, (t, c) in enumerate(zip(d["t"], d["change"]))]
    assert rule_t13_door(_ctx(d)) == []


def test_no_door_features_no_event():
    assert rule_t13_door(_ctx(None)) == []
    assert rule_t13_door(_ctx({"version": 1, "found": False, "reason": "x"})) == []


def test_t13_rule_entry_names_door():
    r = RULES["T13"]
    assert "двер" in r["name"].lower()
    assert r["rule"]["clause"].startswith("п. 8")


# --- живой путь по видео (без кеша признаков двери) ---------------------------------------------

def _live(name: str):
    path = VIDEOS_DIR / f"{name}.mov"
    pj = RESULTS_DIR / "perception" / f"{name}.json"
    if not path.exists() or not pj.exists():
        pytest.skip(f"нет {path} или {pj}")
    from pipeline.door import compute_door
    from pipeline.perception import perception_from_dict
    from pipeline.video import probe

    info = probe(str(path))
    perc = perception_from_dict(json.loads(pj.read_text(encoding="utf-8")), info)
    d = compute_door(str(path), info, perc)
    json.dumps(d)   # только числа/строки — сериализуется
    return d, info


@pytest.mark.parametrize("name,label", [("Х-32", 5.0), ("Х-33", 6.0)])
def test_live_door_left_open(name, label):
    d, info = _live(name)
    assert d["found"]
    ctx = _ctx(d, info.duration_sec)
    ev = rule_t13_door(ctx)
    assert len(ev) == 1 and abs(ev[0].timecode_sec - label) <= 1.5, ev


@pytest.mark.parametrize("name", ["Х-31", "Х-3", "Х-16", "Х-39"])
def test_live_door_closed_clean(name):
    d, info = _live(name)
    assert rule_t13_door(_ctx(d, info.duration_sec)) == []
