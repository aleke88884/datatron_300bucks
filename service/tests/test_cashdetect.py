"""Наличные, сливающиеся с деревом стола (pipeline/cashdetect.py): синтетика + целевые видео датасета.

Синтетика проверяет, что пачка бежевых купюр отличается от дерева по светлоте/насыщенности относительно дерева
в этом же кадре, а белый документ и розовая ткань (кофта кассира) пачкой не считаются; что признак не зависит от
экспозиции (тот же кадр темнее/светлее). Датасет-тесты (по кешу results/) — что Х-26 (T1) и Х-4/Х-47 (T7) находятся,
а чистые ролики с теми же действиями — нет."""
from __future__ import annotations

import json
import sys

import cv2
import numpy as np
import pytest

from .conftest import RESULTS_DIR, SERVICE_DIR, VIDEOS_DIR

if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

from pipeline import cashdetect as cd  # noqa: E402

H, W = 848, 464
DESK = (60, 380, 400, 520)


def _hsv_patch(h, s, v, shape):
    hsv = np.zeros(shape + (3,), np.uint8)
    hsv[..., 0], hsv[..., 1], hsv[..., 2] = h, s, v
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _scene(gain: float = 1.0, stack=True, paper=False, pink=False) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    frame = np.full((H, W, 3), 235, np.uint8)                        # белая стена
    x1, y1, x2, y2 = DESK
    wood = _hsv_patch(15, 75, 195, (y2 - y1, x2 - x1)).astype(np.int16)
    grain = (rng.normal(0, 6, (y2 - y1, 1)) * np.ones((1, x2 - x1)))[..., None]   # продольные волокна
    frame[y1:y2, x1:x2] = np.clip(wood + grain, 0, 255).astype(np.uint8)
    if stack:      # пачка 5000 тг: кремовая, менее насыщенная, светлее дерева, с белыми полосами
        frame[440:452, 200:224] = _hsv_patch(16, 38, 228, (12, 24))
        frame[443:445, 200:224] = _hsv_patch(16, 10, 250, (2, 24))
    if paper:      # белый документ
        frame[400:430, 300:340] = _hsv_patch(0, 4, 250, (30, 40))
    if pink:       # розовая ткань (рукав кассира)
        frame[470:490, 100:140] = _hsv_patch(3, 45, 230, (20, 40))
    frame = np.clip(frame.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    desk = np.zeros((H, W), bool)
    desk[y1:y2, x1:x2] = True
    return frame, desk


def _cands(frame, desk):
    c, s_w, v_w = cd.pale_candidates(frame, desk, np.zeros_like(desk), int(desk.sum()))
    return c, v_w


@pytest.mark.parametrize("gain", [0.85, 1.0, 1.08])
def test_beige_stack_found_under_exposure_drift(gain):
    frame, desk = _scene(gain)
    c, v_w = _cands(frame, desk)
    stack = [o for o in c if o["bbox"][0] < 230 and o["bbox"][2] > 195 and 430 < o["bbox"][1] < 460]
    assert stack, c
    assert cd.is_cashlike(stack[0], v_w), stack[0]
    # на чистом дереве — ни одного кандидата (волокна и блики не «бумага»)
    clean, _ = _cands(*_scene(gain, stack=False))
    assert clean == []


def test_white_balance_restores_stack_under_warm_camera():
    """Камера с тёплым балансом белого: после приведения стен к нейтральному пачка снова опознаётся."""
    frame, desk = _scene()
    warm = np.clip(frame.astype(np.float32) * np.array([0.82, 1.0, 1.06], np.float32), 0, 255).astype(np.uint8)
    g = cd.wb_gains([(0.0, warm)])
    assert g[0] > 1.1 and g[2] < 0.95, g
    c, v_w = _cands(cd._balance(warm, g), desk)
    assert any(cd.is_cashlike(o, v_w) for o in c), c
    # на нейтральной камере коэффициенты ≈ 1
    assert np.allclose(cd.wb_gains([(0.0, frame)]), 1.0, atol=0.03)


def test_stack_is_inside_wood_colour_range():
    """Сама проблема: средний цвет пачки попадает в диапазон «дерева» states.WOOD_HSV."""
    from pipeline.states import WOOD_HSV

    hsv = cv2.cvtColor(_hsv_patch(16, 45, 215, (1, 1)), cv2.COLOR_BGR2HSV)[0, 0]
    assert all(WOOD_HSV[0][k] <= hsv[k] <= WOOD_HSV[1][k] for k in range(3))


def test_document_and_pink_fabric_are_not_cash():
    frame, desk = _scene(stack=False, paper=True, pink=True)
    c, v_w = _cands(frame, desk)
    assert c, "белый документ должен быть найден как бледный предмет"
    assert not any(cd.is_cashlike(o, v_w) for o in c), c


def test_person_mask_hides_candidates():
    frame, desk = _scene()
    pm = np.zeros_like(desk)
    pm[420:470, 180:250] = True
    c, _, _ = cd.pale_candidates(frame, desk, pm, int(desk.sum()))
    assert not any(o["bbox"][0] < 230 and o["bbox"][2] > 195 and 430 < o["bbox"][1] < 460 for o in c)


def _cash_dict(objs_by_t):
    return {"version": cd.CASH_VERSION, "fps": 2.0, "desk": {"bbox": [0, 0, 10, 10], "area": 1000},
            "frames": [{"t": t, "wood_s": 75.0, "wood_v": 195.0, "objs": objs} for t, objs in objs_by_t]}


def _obj(i, seen=True, occluded=False, s_rel=0.55, white=0.0, dhue=1.0):
    return {"id": i, "bbox": [100, 100, 124, 112], "area": 250, "area_rel": 0.01, "age_sec": 0.0, "seen": seen,
            "occluded": occluded, "s_rel": s_rel, "v_rel": 30.0, "hue": 16.0, "dhue": dhue, "fill": 0.8,
            "white": white, "peak": 1.6}


def test_cash_on_grid_persists_through_occlusion_and_needs_static_time():
    ts = [i * 0.5 for i in range(12)]
    rows = []
    for t in ts:
        if t < 1.0:
            rows.append((t, []))
        elif t < 3.0:
            rows.append((t, [_obj(1)]))
        elif t < 4.5:
            rows.append((t, [_obj(1, seen=False, occluded=True)]))   # кассир наклонился над столом
        else:
            rows.append((t, [_obj(1)]))
    pres, boxes, _ = cd.cash_on_grid(_cash_dict(rows), np.asarray(ts))
    assert not pres[:2].any() and pres[2:].all()
    assert boxes[5] == [100, 100, 124, 112]
    # мелькнувший на 1 с предмет — не «лежит»
    short = [(t, [_obj(2)] if 1.0 <= t < 2.0 else []) for t in ts]
    assert not cd.cash_on_grid(_cash_dict(short), np.asarray(ts))[0].any()
    # белый документ — не наличные
    doc = [(t, [_obj(3, s_rel=0.2, white=0.7)]) for t in ts]
    assert not cd.cash_on_grid(_cash_dict(doc), np.asarray(ts))[0].any()


def test_cash_cache_is_numeric_only():
    d = RESULTS_DIR / "cash"
    files = sorted(d.glob("*.json")) if d.exists() else []
    if not files:
        pytest.skip("нет кеша results/cash")

    def walk(x):
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        else:
            assert x is None or isinstance(x, (int, float, bool)), f"не число: {x!r}"[:80]

    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        assert data["version"] == cd.CASH_VERSION
        walk(data)


# --- целевые видео датасета (по кешу признаков, ~1 с на видео) -------------------------------------------

TARGETS = {"Х-26": ("T1", 9.0), "Х-4": ("T7", 57.0), "Х-47": ("T7", None)}
LOOKALIKES = ["Х-14", "Х-43", "Х-45", "Х-5", "Х-48", "Х-3", "Х-10", "Х-29"]


def _analyze(name):
    p = VIDEOS_DIR / f"{name}.mov"
    if not p.exists():
        p = VIDEOS_DIR / f"{name.replace('Х', 'X')}.mov"
    if not p.exists():
        pytest.skip(f"нет видео {name}")
    from pipeline import analyze_video

    return analyze_video(str(p))


@pytest.mark.dataset
@pytest.mark.parametrize("name", list(TARGETS))
def test_camouflaged_cash_targets(name, monkeypatch):
    monkeypatch.setenv("EVIDENCE_FRAMES", "0")
    code, label = TARGETS[name]
    res = _analyze(name)
    assert res["verdict"] == "violation"
    evs = [v for v in res["violations"] if v["type_code"] == code]
    assert evs, res["violations"]
    assert (res["meta"]["scene"] or {}).get("cash_frames", 0) > 0
    if label is not None:
        assert min(abs(v["timecode_sec"] - label) for v in evs) <= 3.0
    assert "купюр" in evs[0]["description"] or "наличные" in evs[0]["description"]


@pytest.mark.dataset
@pytest.mark.parametrize("name", LOOKALIKES)
def test_cash_lookalikes_stay_clean(name, monkeypatch):
    monkeypatch.setenv("EVIDENCE_FRAMES", "0")
    res = _analyze(name)
    assert res["verdict"] == "no_violation", res["violations"]
