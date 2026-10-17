"""Дорожки состояний сцены (pipeline/timeline.py) для шкалы времени UI: формат, пороги как у правил, темнота = нет данных."""
from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from .conftest import RESULTS_DIR, SERVICE_DIR, VIDEOS_DIR

if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

from pipeline.scene import DESK_OBJ_MIN, Scene  # noqa: E402
from pipeline.timeline import build_timeline  # noqa: E402
from pipeline.video import VideoInfo  # noqa: E402


def _scene(n: int = 20, **over) -> Scene:
    """Синтетическая сцена на сетке 2 к/с (10 с)."""
    t = np.arange(n) / 2.0
    z = np.zeros(n, bool)
    base = dict(
        info=VideoInfo(path="synthetic.mov", fps=30.0, frame_count=n * 15, width=464, height=848, duration_sec=n / 2.0),
        t=t, cashier_inside=z.copy(), cashier_at_desk=z.copy(), cashier_h=np.zeros(n), cashier_center=[None] * n,
        n_inside=np.zeros(n, int), n_strong=np.zeros(n, int), n_visitors=np.zeros(n, int), visitor_in_work_zone=z.copy(),
        visitor_ids=[[] for _ in range(n)], screen=[None] * n, lights_off=z.copy(), bag_on_desk=z.copy(),
        bag_area=np.zeros(n), desk_obj_area=np.zeros(n), desk_objs=[[] for _ in range(n)], safe_open_score=[None] * n,
        roles={}, anchors={"desk": {"bbox": [0, 0, 1, 1]}, "safe": {"bbox": [0, 0, 1, 1]}, "laptop": {"bbox": [0, 0, 1, 1]}},
    )
    base.update(over)
    return Scene(**base)


def test_format_and_intervals():
    n = 20
    cash = np.zeros(n, bool)
    cash[0:8] = True                       # 0–4 с
    cash[9:12] = True                      # провал в одну ячейку (4.0–4.5) заполняется → 0–6 с
    vis = np.zeros(n, int)
    vis[10:14] = 1
    vis[14:18] = 2
    tl = build_timeline(_scene(cashier_inside=cash, cashier_at_desk=cash.copy(), n_visitors=vis))
    assert tl["step"] == 0.5 and tl["duration"] == 10.0
    assert tl["lanes"]["cashier"] == [[0.0, 6.0]]
    assert tl["lanes"]["visitors"] == [[5.0, 7.0, 1], [7.0, 9.0, 2]]
    assert set(tl["lanes"]) >= {"cashier", "at_desk", "visitors", "safe_open", "safe_unknown", "desk_items", "bag",
                                "desk_unknown", "screen_active", "screen_locked", "screen_unknown", "lights_off"}
    assert tl["anchors"] == {"desk": True, "safe": True, "laptop": True}
    json.dumps(tl)   # только числа/списки — сериализуется в ответ API


def test_thresholds_match_rules():
    """Сейф «открыт» — как в T2 (score ≥ 1.0); предметы — как в T1/T7 (≥ DESK_OBJ_MIN или сумка)."""
    n = 20
    score = [None] * 4 + [0.5] * 6 + [1.0] * 4 + [1.5] * 2 + [0.2] * 4
    desk = np.zeros(n)
    desk[2:6] = DESK_OBJ_MIN
    desk[6:8] = DESK_OBJ_MIN - 1
    tl = build_timeline(_scene(safe_open_score=score, desk_obj_area=desk))
    assert tl["lanes"]["safe_open"] == [[5.0, 8.0]]
    assert tl["lanes"]["safe_unknown"] == [[0.0, 2.0]]          # None — «нет данных», а не «закрыт»
    assert tl["lanes"]["desk_items"] == [[1.0, 3.0]]


def test_screen_states_and_unknown():
    scr = [None] * 4 + ["active"] * 6 + ["off"] * 4 + ["locked"] * 6
    tl = build_timeline(_scene(screen=scr))
    assert tl["lanes"]["screen_unknown"] == [[0.0, 2.0]]
    assert tl["lanes"]["screen_active"] == [[2.0, 5.0]]
    assert tl["lanes"]["screen_locked"] == [[7.0, 10.0]]
    # ноутбук не найден в сцене: дорожка не «неизвестно на всём ролике», а пустая + anchors.laptop = False
    tl2 = build_timeline(_scene(screen=scr, anchors={"desk": None, "safe": None, "laptop": None}))
    assert tl2["lanes"]["screen_unknown"] == [] and tl2["anchors"]["laptop"] is False


def test_darkness_is_unreliable_not_negative():
    """В темноте состояния по изображению не определяются: дорожки пусты, сейф/экран/стол — «нет данных»."""
    n = 20
    dark = np.zeros(n, bool)
    dark[12:] = True                        # 6–10 с
    on = np.ones(n, bool)
    tl = build_timeline(_scene(lights_off=dark, cashier_inside=on, cashier_at_desk=on, screen=["active"] * n,
                               safe_open_score=[1.2] * n, bag_on_desk=on))
    L = tl["lanes"]
    assert L["lights_off"] == [[6.0, 10.0]] and tl["unreliable"] == [[6.0, 10.0]]
    assert L["cashier"] == [[0.0, 6.0]] and L["screen_active"] == [[0.0, 6.0]] and L["safe_open"] == [[0.0, 6.0]]
    assert L["bag"] == [[0.0, 6.0]]
    for k in ("safe_unknown", "screen_unknown", "desk_unknown"):
        assert L[k] == [[6.0, 10.0]], k


def test_empty_scene():
    tl = build_timeline(_scene(n=0, info=VideoInfo(path="synthetic.mov", fps=30.0, frame_count=0, width=464, height=848, duration_sec=0.0)))
    assert all(v == [] for v in tl["lanes"].values())


@pytest.mark.parametrize("name", ["Х-11", "Х-3"])
def test_timeline_on_cached_features(name, monkeypatch):
    """На реальных признаках (кеш results/): интервалы в пределах ролика, упорядочены, ответ компактный."""
    video = VIDEOS_DIR / f"{name}.mov"
    if not video.exists() or not (RESULTS_DIR / "perception" / f"{name}.json").exists():
        pytest.skip("нет видео или кеша признаков")
    monkeypatch.setenv("FEATURES_CACHE", str(RESULTS_DIR))
    monkeypatch.setenv("EVIDENCE_FRAMES", "0")
    from pipeline import analyze_video

    res = analyze_video(str(video))
    tl = res["meta"]["timeline"]
    assert tl is not None and abs(tl["duration"] - res["meta"]["duration_sec"]) <= 0.1
    assert len(json.dumps(tl)) < 20_000
    for key, ivs in tl["lanes"].items():
        prev = -1.0
        for iv in ivs:
            assert 0.0 <= iv[0] < iv[1] <= tl["duration"] + 0.05, (key, iv)
            assert iv[0] >= prev, key
            prev = iv[1]
    if name == "Х-11":
        # свет гаснет в момент T3 (≈16 с) — дорожка «свет выключен» и «нет данных» начинаются там же
        t3 = [v for v in res["violations"] if v["type_code"] == "T3"][0]
        assert abs(tl["lanes"]["lights_off"][0][0] - t3["timecode_sec"]) <= 1.0
        assert tl["unreliable"] == tl["lanes"]["lights_off"]
