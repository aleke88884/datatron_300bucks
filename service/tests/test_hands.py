"""Слой позы (pipeline.pose) и правила по рукам (pipeline.rules_hands) на синтетических данных — без датасета."""
from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from .conftest import SERVICE_DIR

if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

from pipeline import pose as P  # noqa: E402
from pipeline import rules_hands as RH  # noqa: E402
from pipeline.perception import PersonObs, Track  # noqa: E402

FPS = 3.0
DUR = 12.0


def _kpts(wr_left=(80.0, 150.0), wr_right=(120.0, 150.0), wrist_score=0.9) -> list:
    k = np.zeros((17, 3))
    k[:, 2] = 0.9
    k[0] = (100, 70, 0.9)
    k[P.L_SH] = (80, 100, 0.9)
    k[P.R_SH] = (120, 100, 0.9)
    k[P.L_EL] = (75, 130, 0.9)
    k[P.R_EL] = (125, 130, 0.9)
    k[P.L_WR] = (*wr_left, wrist_score)
    k[P.R_WR] = (*wr_right, wrist_score)
    k[P.L_HIP] = (85, 180, 0.9)
    k[P.R_HIP] = (115, 180, 0.9)
    return k.tolist()


def _ctx(raised: tuple[float, float] | None, stand_at: float = 6.0, leave_at: float = 9.0, pose_on: bool = True):
    """Кассир (трек 1) сидит у стола весь ролик; посетитель (трек 2) сидит, встаёт в stand_at и уходит в leave_at.
    raised=(t0, t1) — интервал, когда правая кисть кассира вынесена далеко от позы покоя."""
    times = list(np.round(np.arange(0.0, DUR, 1.0 / FPS), 3))
    cashier = Track(id=1)
    visitor = Track(id=2)
    poses = []
    for t in times:
        cashier.obs.append(PersonObs(t=t, xyxy=(60, 60, 140, 220), score=0.9, inside=True, track_id=1))
        if t <= leave_at:
            h = 100.0 if t < stand_at else 180.0
            visitor.obs.append(PersonObs(t=t, xyxy=(250, 300 - h, 320, 300), score=0.9, inside=True, track_id=2))
        right = (180.0, 40.0) if raised and raised[0] <= t <= raised[1] else (120.0, 150.0)
        poses.append([{"track_id": 1, "kpts": _kpts(wr_right=right)}])
    grid = np.arange(0.0, DUR, 0.5)
    scene = SimpleNamespace(
        roles={1: "cashier", 2: "visitor"}, cashier_track_ids=[1], t=grid,
        cashier_at_desk=np.ones(len(grid), bool), info=SimpleNamespace(duration_sec=DUR),
        idx=lambda t: int(np.argmin(np.abs(grid - t))),
    )
    pose = {"version": P.POSE_VERSION, "times": times, "poses": poses} if pose_on else None
    return SimpleNamespace(scene=scene, tracks=[cashier, visitor], cache={"pose": pose}, info=scene.info)


def test_box_transform_keeps_model_aspect():
    cx, cy, w, h = P.PoseEstimator._box_transform((10, 20, 50, 220))
    assert (cx, cy) == (30.0, 120.0)
    assert w / h == pytest.approx(P.INPUT_W / P.INPUT_H)
    assert h >= 200 * P.BOX_PADDING - 1e-6


def test_body_frame_is_scale_invariant():
    k = np.asarray(_kpts(wr_right=(180, 40)))
    k2 = k.copy()
    k2[:, :2] *= 3.0
    s1 = P.track_series({"times": [0.0], "poses": [[{"track_id": 1, "kpts": k.tolist()}]]}, [1])
    s2 = P.track_series({"times": [0.0], "poses": [[{"track_id": 1, "kpts": k2.tolist()}]]}, [1])
    np.testing.assert_allclose(s1["wrists_rel"], s2["wrists_rel"], atol=1e-6)


def test_invisible_wrist_is_unknown_not_zero():
    k = _kpts(wrist_score=0.1)
    s = P.track_series({"times": [0.0], "poses": [[{"track_id": 1, "kpts": k}]]}, [1])
    assert np.isnan(s["wrists_rel"]).all()


def test_departure_detected_from_visitor_height():
    deps = RH.visitor_departures(_ctx(None))
    assert len(deps) == 1 and deps[0]["track_id"] == 2
    assert deps[0]["t_stand"] == pytest.approx(6.0, abs=0.4)


def test_t6_fires_when_cashier_works_after_client_stands_up():
    evs = RH.rule_t6_hands(_ctx(raised=(6.6, 10.0)))
    assert len(evs) == 1
    ev = evs[0]
    assert ev.type_code == "T6" and 6.0 <= ev.timecode_sec <= 7.5
    assert ev.confidence < 0.7, "индикатор по позе — пониженная уверенность"
    assert "требует проверки" in ev.description and "Признак" in ev.description
    assert len(ev.evidence_times) == 3


def test_t6_silent_when_hands_at_rest_after_client_leaves():
    assert RH.rule_t6_hands(_ctx(raised=None)) == []


def test_t6_silent_for_short_reach_or_reach_before_standing():
    assert RH.rule_t6_hands(_ctx(raised=(6.6, 7.6))) == []      # < 2 с
    assert RH.rule_t6_hands(_ctx(raised=(2.0, 5.5))) == []      # пока клиент сидит — штатная работа при клиенте


def test_t6_silent_when_cashier_reaches_into_safe():
    """Кассир после ухода клиента убирает деньги в сейф (Х-39) — штатно, не T6."""
    ctx = _ctx(raised=(6.6, 10.0))
    ctx.scene.anchors = {"safe": {"bbox": (160, 20, 240, 120)}}
    assert RH.rule_t6_hands(ctx) == []
    ctx.scene.anchors = {"safe": {"bbox": (300, 300, 400, 400)}}   # сейф в другом месте — рука у оборудования
    assert len(RH.rule_t6_hands(ctx)) == 1


def test_t6_silent_without_pose():
    assert RH.rule_t6_hands(_ctx(raised=(6.6, 10.0), pose_on=False)) == []


def test_pose_model_runs_on_frame_if_present():
    est = P.get_estimator()
    if not est.available():
        pytest.skip("нет models/rtmpose_m_body7.onnx (scripts/get_models.py)")
    frame = np.full((400, 300, 3), 127, np.uint8)
    out = est(frame, [(50, 50, 150, 350), (160, 60, 260, 380)])
    assert len(out) == 2
    for k in out:
        assert k.shape == (17, 3)
        assert np.isfinite(k).all()
