"""Самокалибровка камеры (pipeline/camera.py) и преобразования харнесса устойчивости (scripts/robustness.py).

Проверяется, что: (1) на эталонной камере обучающего датасета калибровка ничего не меняет (режим «fixed»,
масштаб H/848, экспозиция 1.0) — результаты офлайн-оценки остаются прежними; (2) на кадре с другой геометрией
(зеркало, другое разрешение, другая экспозиция) пороги пересчитываются, а тест «внутри» переключается на
относительный, не завязанный на координаты двери.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from .conftest import RESULTS_DIR, SERVICE_DIR

if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))
if str(SERVICE_DIR / "scripts") not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR / "scripts"))

from pipeline import camera  # noqa: E402
from pipeline.perception import Perception, PersonObs, perception_from_dict  # noqa: E402
from pipeline.video import VideoInfo  # noqa: E402


def _info(w=464, h=848) -> VideoInfo:
    return VideoInfo(path="x", width=w, height=h, fps=30.0, frame_count=300, duration_sec=10.0)


def _perc(boxes_by_frame: list[list[tuple]], info: VideoInfo, inside=None) -> Perception:
    """boxes_by_frame: [[(track_id, x1, y1, x2, y2, score), ...], ...] на 3 к/с; inside — функция фиксированной геометрии."""
    persons = []
    for i, frame in enumerate(boxes_by_frame):
        ps = []
        for tid, x1, y1, x2, y2, sc in frame:
            ins = bool(inside((x1, y1, x2, y2))) if inside else True
            ps.append(PersonObs(t=i / 3.0, xyxy=(x1, y1, x2, y2), score=sc, inside=ins, track_id=tid, state=1 if ins else -1))
        persons.append(ps)
    return Perception(info=info, times=[i / 3.0 for i in range(len(persons))], persons=persons,
                      objects=[[] for _ in persons], tracks=[])


DESK = {"bbox": [96, 372, 344, 482], "center": [206.0, 415.0], "axis": [0.97, -0.24]}


# ---------------------------------------------------------------- экспозиция и масштаб

def test_exposure_gain_deadzone_and_clip():
    assert camera.exposure_gain(137.5) == 1.0
    assert camera.exposure_gain(125.0) == 1.0          # разброс обучающих роликов — без коррекции
    assert camera.exposure_gain(150.0) == 1.0
    assert camera.exposure_gain(90.0) == pytest.approx(137.5 / 90.0)
    assert camera.exposure_gain(20.0) == camera.EXPOSURE_CLIP[1]
    assert camera.exposure_gain(None) == 1.0


def test_exposure_level_ignores_black_borders():
    g = np.full((100, 100), 140, np.uint8)
    g[:, :40] = 0                                        # поля альбомного кадра / углы после поворота
    assert camera.exposure_level(g) == pytest.approx(140.0)
    assert camera.exposure_gain(camera.exposure_level(g)) == 1.0


def test_apply_gain_hsv_changes_only_value():
    hsv = np.zeros((2, 2, 3), np.uint8)
    hsv[..., 0], hsv[..., 1], hsv[..., 2] = 20, 100, 100
    out = camera.apply_gain_hsv(hsv, 1.5)
    assert (out[..., 0] == 20).all() and (out[..., 1] == 100).all() and (out[..., 2] == 150).all()
    assert camera.apply_gain_hsv(hsv, 1.0) is hsv


def test_scene_scale_reference_and_override():
    info = _info()
    # эталонная камера: люди ростом ~0.23 кадра → масштаб 1.0 (H/848)
    p = _perc([[(1, 150, 200, 250, 200 + 0.23 * 848, 0.9)] for _ in range(12)], info)
    assert camera.scene_scale(p, info) == pytest.approx(1.0)
    # та же камера, перекодированная в 720p по высоте — масштаб по H
    info720 = _info(394, 720)
    p720 = _perc([[(1, 100, 100, 180, 100 + 0.23 * 720, 0.9)] for _ in range(12)], info720)
    assert camera.scene_scale(p720, info720) == pytest.approx(720 / 848)
    # альбомный кадр 1280×720, люди крупнее (×1.6 к H/848) — масштаб по росту людей
    infoL = _info(1280, 720)
    pL = _perc([[(1, 500, 100, 620, 100 + 0.37 * 720, 0.9)] for _ in range(12)], infoL)
    s = camera.scene_scale(pL, infoL)
    assert s == pytest.approx(0.37 * 720 / camera.REF_PERSON_H, rel=1e-3)
    # мало наблюдений — без переопределения
    assert camera.scene_scale(_perc([[(1, 0, 0, 10, 400, 0.9)]], infoL), infoL) == pytest.approx(720 / 848)


# ---------------------------------------------------------------- тест «внутри»

def test_relative_state_bands():
    href = 200.0
    cy = 415.0
    assert camera.relative_state((0, 100, 50, 260), cy, href) == -1          # 0.78 роста выше центра стола — за стеклом
    assert camera.relative_state((0, 200, 50, 400), cy, href) == 1           # на уровне стола — внутри
    assert camera.relative_state((0, 160, 50, 300), cy, href) == 0           # спорная полоса, фигура мелкая
    assert camera.relative_state((0, 120, 50, 300), cy, href) == 1           # спорная полоса, фигура крупная (ближе стекла)


def _fixed_like(xyxy) -> bool:
    """Упрощённая «геометрия эталонной камеры»: слева-вверху стекло, правее 0.55 ширины — всегда внутри."""
    cx = (xyxy[0] + xyxy[2]) / 2
    return cx >= 0.55 * 464 or xyxy[3] >= 345


def _scene_frames(mirror: bool) -> list[list[tuple]]:
    """Кассир у стола + человек за стеклом (низ рамки высоко над столом). mirror — зеркальный кадр."""
    frames = []
    for _ in range(30):
        cashier = (1, 160, 200, 260, 400, 0.9)
        outside = (2, 60, 160, 110, 260, 0.75)
        if mirror:
            cashier = (1, 464 - 260, 200, 464 - 160, 400, 0.9)
            outside = (2, 464 - 110, 160, 464 - 60, 260, 0.75)
        frames.append([cashier, outside])
    return frames


def test_calibrate_keeps_fixed_geometry_on_reference_camera(monkeypatch):
    monkeypatch.delenv("CAMERA_INSIDE", raising=False)
    info = _info()
    p = _perc(_scene_frames(False), info, inside=_fixed_like)
    cam = camera.calibrate(p, {"scene": {"desk": DESK, "median_brightness": 138.0}}, info)
    assert cam["mode"] == "fixed" and cam["scale"] == 1.0 and cam["gain"] == 1.0
    assert all(o.inside == (o.track_id == 1) for ps in p.persons for o in ps)


def test_calibrate_switches_to_relative_on_mirrored_camera(monkeypatch):
    monkeypatch.delenv("CAMERA_INSIDE", raising=False)
    info = _info()
    mirrored_desk = dict(DESK, center=[464 - 206.0, 415.0], bbox=[464 - 344, 372, 464 - 96, 482])
    p = _perc(_scene_frames(True), info, inside=_fixed_like)
    # фиксированная геометрия ошибочно считает человека за стеклом «внутри» (он теперь справа)
    assert all(o.inside for ps in p.persons for o in ps)
    cam = camera.calibrate(p, {"scene": {"desk": mirrored_desk, "median_brightness": 138.0}}, info)
    assert cam["mode"] == "relative"
    for ps in p.persons:
        for o in ps:
            assert o.inside == (o.track_id == 1)


def test_calibrate_mode_override(monkeypatch):
    info = _info()
    monkeypatch.setenv("CAMERA_INSIDE", "fixed")
    p = _perc(_scene_frames(True), info, inside=_fixed_like)
    assert camera.calibrate(p, {"scene": {"desk": DESK}}, info)["mode"] == "fixed"
    monkeypatch.setenv("CAMERA_INSIDE", "relative")
    p = _perc(_scene_frames(True), info, inside=_fixed_like)
    assert camera.calibrate(p, {"scene": {}}, info)["mode"] == "all-inside"


@pytest.mark.dataset
def test_reference_camera_videos_stay_in_fixed_mode(monkeypatch):
    """На всех 42 роликах эталонной камеры калибровка не меняет режим, масштаб и экспозицию."""
    monkeypatch.delenv("CAMERA_INSIDE", raising=False)
    states = sorted((RESULTS_DIR / "states").glob("[XХ]-*.json"))
    if not states:
        pytest.skip("нет кеша признаков results/states")
    for sp in states:
        st = json.loads(sp.read_text(encoding="utf-8"))
        sc = st["scene"]
        pp = RESULTS_DIR / "perception" / sp.name
        if not pp.exists():
            continue
        info = VideoInfo(sp.stem, sc["width"], sc["height"], sc["fps"], 0, sc["duration_sec"])
        perc = perception_from_dict(json.loads(pp.read_text(encoding="utf-8")), info)
        cam = camera.calibrate(perc, st, info)
        assert cam["mode"] == "fixed", (sp.stem, cam)
        assert cam["scale"] == pytest.approx(sc["height"] / 848.0), (sp.stem, cam)
        assert cam["gain"] == 1.0, (sp.stem, cam)


# ---------------------------------------------------------------- харнесс

def test_robustness_transforms_shapes(tmp_path):
    import cv2
    from robustness import TRANSFORMS, make_variant

    f = np.random.default_rng(0).integers(0, 255, (848, 464, 3), dtype=np.uint8)
    for k, fn in TRANSFORMS.items():
        g = fn(f)
        assert g.dtype == np.uint8 and g.ndim == 3, k
        assert g.shape[:2] == ((720, 1280) if k == "land" else (848, 464)), k
    src = tmp_path / "src.mp4"
    wr = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (464, 848))
    for _ in range(5):
        wr.write(f)
    wr.release()
    dst = make_variant(src, "land", tmp_path / "v" / "src__land.mp4")
    cap = cv2.VideoCapture(str(dst))
    ok, fr = cap.read()
    cap.release()
    assert ok and fr.shape[:2] == (720, 1280)
