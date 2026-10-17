"""T8 (личные вещи, еда, напитки на рабочем месте): логика pipeline/personal_items.py на синтетических детекциях
и, при наличии датасета, на Х-40 (сумка) / Х-46 (стакан) с проверкой таймкода против разметки."""
from __future__ import annotations

import sys

import numpy as np
import pytest

from .conftest import SERVICE_DIR, VIDEOS_DIR

if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

from pipeline import personal_items as pi  # noqa: E402
from pipeline.context import VideoContext  # noqa: E402
from pipeline.detect import Detection  # noqa: E402
from pipeline.perception import Perception, PersonObs  # noqa: E402
from pipeline.video import VideoInfo  # noqa: E402

INFO = VideoInfo(path="synthetic.mov", width=464, height=848, fps=30.0, frame_count=900, duration_sec=30.0)
DESK = [60, 370, 320, 500]
FPS = 3.0


class FakeScene:
    """Минимальная сцена: сетка 2 к/с, кассир внутри и у стола, высота рамки кассира задаётся рядом."""

    def __init__(self, cashier_h: np.ndarray, roles: dict):
        self.info = INFO
        self.t = np.arange(len(cashier_h)) / 2.0
        self.cashier_h = cashier_h
        self.cashier_at_desk = np.ones(len(cashier_h), bool)
        self.cashier_inside = np.ones(len(cashier_h), bool)
        self.roles = roles
        self.anchors = {"desk": {"bbox": DESK}}


def _perception(objects_at: dict[int, list[Detection]], persons_at: dict[int, list[PersonObs]], n: int = 90) -> Perception:
    times = [i / FPS for i in range(n)]
    return Perception(info=INFO, times=times, persons=[persons_at.get(i, []) for i in range(n)],
                      objects=[objects_at.get(i, []) for i in range(n)], tracks=[])


def _cashier(i: int, x: float = 80.0) -> PersonObs:
    return PersonObs(t=i / FPS, xyxy=(x, 250.0, x + 70.0, 470.0), score=0.9, inside=True, track_id=1)


def _ctx(perc: Perception, scene: FakeScene) -> VideoContext:
    ctx = VideoContext(path="synthetic.mov", info=INFO, bright_t=np.zeros(0), bright_v=np.zeros(0))
    ctx.scene = scene
    ctx.cache["perception"] = perc
    return ctx


@pytest.fixture(autouse=True)
def _no_video(request, monkeypatch):
    """Без видео: пиксельные признаки подменяются; кеш признаков не пишется (кроме тестов на датасете)."""
    if request.node.get_closest_marker("dataset"):
        return
    monkeypatch.delenv("FEATURES_CACHE", raising=False)
    monkeypatch.setattr(pi.PixelMemo, "green_frac", lambda self, t, box: 0.0)
    monkeypatch.setattr(pi.PixelMemo, "crop_detect", lambda self, t, roi, classes: [])


def _standing_then_sitting(n: int = 60, sit_at: int = 36) -> np.ndarray:
    h = np.full(n, 200.0)
    h[sit_at:] = 120.0
    return h


def _carried_bag(start: int = 15, k: int = 6) -> tuple[dict, dict]:
    objs, persons = {}, {}
    for j in range(start, start + k):
        x = 80.0 + 6.0 * (j - start)              # сумка движется вместе с кассиром
        persons[j] = [_cashier(j, x)]
        objs[j] = [Detection("handbag", 0.7, (x + 5, 290.0, x + 35, 360.0))]
    return objs, persons


def test_bag_carried_by_cashier_gives_t8_at_sit_down():
    objs, persons = _carried_bag()
    perc = _perception(objs, persons)
    sc = FakeScene(_standing_then_sitting(), {1: "cashier"})
    evs = pi.rule_t8(_ctx(perc, sc))
    assert len(evs) == 1 and evs[0].type_code == "T8" and evs[0].extras["kind"] == "bag"
    assert evs[0].timecode_sec == pytest.approx(18.0)        # кассир садится (индекс 36 сетки 2 к/с)
    assert "сумка" in evs[0].description and "инкассатор" in evs[0].description


def test_bag_without_sitting_reports_first_sighting():
    objs, persons = _carried_bag()
    perc = _perception(objs, persons)
    sc = FakeScene(np.full(60, 200.0), {1: "cashier"})
    evs = pi.rule_t8(_ctx(perc, sc))
    assert len(evs) == 1 and evs[0].timecode_sec == pytest.approx(5.0)


def test_green_collection_bag_is_not_personal(monkeypatch):
    monkeypatch.setattr(pi.PixelMemo, "green_frac", lambda self, t, box: 0.6)
    objs, persons = _carried_bag()
    assert pi.rule_t8(_ctx(_perception(objs, persons), FakeScene(_standing_then_sitting(), {1: "cashier"}))) == []


def test_bag_carried_by_visitor_is_ignored():
    objs, persons = _carried_bag()
    sc = FakeScene(_standing_then_sitting(), {1: "visitor"})
    assert pi.rule_t8(_ctx(_perception(objs, persons), sc)) == []


def test_static_interior_bag_is_ignored():
    objs, persons = {}, {}
    for j in range(15, 30):
        persons[j] = [_cashier(j)]
        objs[j] = [Detection("handbag", 0.7, (40.0, 400.0, 115.0, 485.0))]   # неподвижная рамка (стул/урна)
    objs[3] = [Detection("handbag", 0.4, (40.0, 401.0, 115.0, 486.0))]       # та же «сумка» без людей рядом
    assert pi.rule_t8(_ctx(_perception(objs, persons), FakeScene(_standing_then_sitting(), {1: "cashier"}))) == []


def test_weak_or_rare_bag_is_ignored():
    objs, persons = _carried_bag(k=2)
    assert pi.rule_t8(_ctx(_perception(objs, persons), FakeScene(_standing_then_sitting(), {1: "cashier"}))) == []
    objs, persons = _carried_bag()
    for j in objs:
        objs[j] = [Detection("handbag", 0.4, objs[j][0].xyxy)]
    assert pi.rule_t8(_ctx(_perception(objs, persons), FakeScene(_standing_then_sitting(), {1: "cashier"}))) == []


def _drinks(scores: list[float], start: int = 39) -> dict:
    return {start + k: [Detection("cup", s, (165.0, 380.0, 180.0, 400.0))] for k, s in enumerate(scores)}


def test_drink_on_desk_with_onset_refinement(monkeypatch):
    # полный кадр видит стакан с 13 с; на кропе зоны стола он виден уже с 11 с
    def crop(self, t, roi, classes):
        return [("cup", 0.8, (152.0, 372.0, 164.0, 388.0))] if t >= 10.9 else []
    monkeypatch.setattr(pi.PixelMemo, "crop_detect", crop)
    perc = _perception(_drinks([0.6, 0.4, 0.55, 0.35, 0.62]), {})
    evs = pi.rule_t8(_ctx(perc, FakeScene(np.full(60, 110.0), {1: "cashier"})))
    assert len(evs) == 1 and evs[0].extras["kind"] == "drink"
    assert evs[0].timecode_sec == pytest.approx(11.0)
    assert evs[0].extras["t_first_full"] == pytest.approx(13.0)


def test_drink_needs_several_confident_detections():
    # как в чистом Х-19: много слабых детекций «стакана» на месте предмета интерьера и одна уверенная
    perc = _perception(_drinks([0.35, 0.38, 0.31, 0.57, 0.33, 0.36]), {})
    assert pi.rule_t8(_ctx(perc, FakeScene(np.full(60, 110.0), {1: "cashier"}))) == []


def test_drink_outside_desk_zone_is_ignored():
    objs = {39 + k: [Detection("cup", 0.7, (400.0, 100.0, 420.0, 130.0))] for k in range(6)}
    assert pi.rule_t8(_ctx(_perception(objs, {}), FakeScene(np.full(60, 110.0), {1: "cashier"}))) == []


def test_desk_zone_is_relative_to_desk():
    z = pi.desk_zone([100, 400, 300, 500], 464, 848)
    assert z == pytest.approx((80.0, 320.0, 320.0, 510.0))
    assert pi.desk_zone(None, 464, 848) is None


def test_pixel_memo_roundtrip(tmp_path, monkeypatch):
    """Пиксельные признаки пишутся в results/personal_items/<видео>.json (только числа) и читаются без видео."""
    monkeypatch.undo()
    monkeypatch.setenv("FEATURES_CACHE", str(tmp_path))
    m = pi.PixelMemo(str(tmp_path / "X-99.mov"), INFO)
    m.data["g|1.00|1,2,3,4"] = 0.25
    m.dirty = True
    m.save()
    f = tmp_path / "personal_items" / "Х-99.json"
    assert f.exists()
    m2 = pi.PixelMemo(str(tmp_path / "X-99.mov"), INFO)
    assert m2.green_frac(1.0, (1, 2, 3, 4)) == 0.25
    assert m2.green_frac(2.0, (1, 2, 3, 4)) is None        # видео нет — признак неизвестен, без исключений


# --- датасет -------------------------------------------------------------------------------------------

@pytest.mark.dataset
@pytest.mark.parametrize("name,label_t", [("Х-40", 17.0), ("Х-46", 11.0)])
def test_t8_on_dataset(name, label_t):
    paths = [p for p in VIDEOS_DIR.glob("*.mov") if p.stem.replace("X", "Х") == name] if VIDEOS_DIR.exists() else []
    if not paths:
        pytest.skip("нет видео датасета")
    from pipeline import analyze_video

    res = analyze_video(str(paths[0]))
    t8 = [v for v in res["violations"] if v["type_code"] == "T8"]
    assert res["verdict"] == "violation" and t8
    assert min(abs(v["timecode_sec"] - label_t) for v in t8) <= 3.0
    assert t8[0]["rule"]["clause"] == "п. 12" and "adilet.zan.kz" in t8[0]["rule"]["url"]
