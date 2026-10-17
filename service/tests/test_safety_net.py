"""Страховочная сетка (pipeline/safety_net.py) и бэкенд llama.cpp (pipeline/vlm.py) — без модели и без видео.

Проверяется логика ворот: событие добавляется только при подтверждении детектором (≥ 2 детекции), моделью
(«да» на ≥ 2 кадрах через ≥ 1 с), при «нет» на опровержениях и «нет» в контрольный момент; любые сомнения → нет.
"""
from __future__ import annotations

import sys

import numpy as np
import pytest

from .conftest import SERVICE_DIR

if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

import pipeline.safety_net as sn  # noqa: E402
import pipeline.vlm as vlm  # noqa: E402
from pipeline.context import Event, VideoContext  # noqa: E402
from pipeline.detect import Detection  # noqa: E402
from pipeline.perception import Perception, PersonObs  # noqa: E402
from pipeline.rules import RULES  # noqa: E402
from pipeline.scene import GRID_FPS, Scene  # noqa: E402
from pipeline.video import VideoInfo  # noqa: E402

INFO = VideoInfo(path="/nonexistent.mov", width=464, height=848, fps=30.0, frame_count=600, duration_sec=20.0)
DESK = {"bbox": [60.0, 360.0, 320.0, 500.0], "center": [190.0, 430.0], "axis": [1.0, 0.0]}


def make_scene(cashier_inside, visitor_ids, lights_off=None, roles=None) -> Scene:
    n = len(cashier_inside)
    t = np.arange(n) / GRID_FPS
    ci = np.asarray(cashier_inside, bool)
    nv = np.asarray([len(v) for v in visitor_ids], int)
    z = np.zeros(n)
    return Scene(
        info=INFO, t=t, cashier_inside=ci, cashier_at_desk=ci.copy(), cashier_h=z.copy(), cashier_center=[None] * n,
        n_inside=nv + ci, n_strong=nv + ci, n_visitors=nv, visitor_in_work_zone=np.zeros(n, bool),
        visitor_ids=[list(v) for v in visitor_ids], screen=[None] * n,
        lights_off=np.zeros(n, bool) if lights_off is None else np.asarray(lights_off, bool),
        bag_on_desk=np.zeros(n, bool), bag_area=z.copy(), desk_obj_area=z.copy(), desk_objs=[[] for _ in range(n)],
        safe_open_score=[None] * n, roles=roles or {1: "cashier", 2: "visitor"}, anchors={"desk": DESK, "safe": None, "laptop": None},
    )


# ----------------------------------------------------------------------------- структурный слой S1


def test_s1_visitor_left_alone_fires_t4():
    # 0–5 с кассир и посетитель вместе, с 5 с кассир вышел, посетитель один до 10 с
    ci = [True] * 10 + [False] * 10
    vis = [[2]] * 20
    evs = sn.structural_events(make_scene(ci, vis), [])
    assert len(evs) == 1
    e = evs[0]
    assert e.type_code == "T4" and e.extras["source"] == sn.SOURCE
    assert abs(e.timecode_sec - 5.0) <= 0.6
    assert "один" in e.description and e.confidence >= 0.5
    assert 1 <= len(e.evidence_times) <= 3


def test_s1_needs_visitor_present_together_with_cashier():
    # посетитель появляется только после ухода кассира (например, кассир вернулся новым треком) — не событие
    ci = [True] * 10 + [False] * 10
    vis = [[]] * 10 + [[2]] * 10
    assert sn.structural_events(make_scene(ci, vis), []) == []


def test_s1_short_absence_and_darkness_ignored():
    ci = [True] * 10 + [False] * 4 + [True] * 6        # 2 с — меньше S1_ALONE_SEC
    assert sn.structural_events(make_scene(ci, [[2]] * 20), []) == []
    ci = [True] * 10 + [False] * 10
    dark = [False] * 10 + [True] * 10                  # в темноте детекции ненадёжны
    assert sn.structural_events(make_scene(ci, [[2]] * 20, lights_off=dark), []) == []


def test_s1_skipped_when_t4_already_found():
    ci = [True] * 10 + [False] * 10
    t4 = Event(type_code="T4", timecode_sec=3.0, description="x" * 30)
    assert sn.structural_events(make_scene(ci, [[2]] * 20), [t4]) == []


# ----------------------------------------------------------------------------- предложения детектора


def make_perc(items_at: dict[float, list[Detection]], duration=20.0, cashier_box=(150, 250, 260, 480), visitor_box=None) -> Perception:
    times = list(np.round(np.arange(0, duration, 1 / 3.0), 3))
    persons, objects = [], []
    for t in times:
        ps = [PersonObs(t=t, xyxy=cashier_box, score=0.9, inside=True, track_id=1)]
        if visitor_box is not None:
            ps.append(PersonObs(t=t, xyxy=visitor_box, score=0.9, inside=True, track_id=2))
        persons.append(ps)
        objects.append([d for tt, ds in items_at.items() if abs(tt - t) < 1e-3 for d in ds])
    return Perception(info=INFO, times=times, persons=persons, objects=objects, tracks=[])


def _t(perc, k):
    return perc.times[k]


def test_proposals_group_repeated_detections_near_cashier():
    base = make_perc({})
    phone = Detection("cell phone", 0.5, (190.0, 330.0, 210.0, 350.0))
    perc = make_perc({_t(base, k): [phone] for k in (30, 31, 33, 36)})
    props = sn.item_proposals(make_scene([True] * 40, [[]] * 40), perc)
    assert len(props) == 1 and props[0].probe == "phone" and props[0].n == 4


def test_single_detection_is_not_a_proposal():
    base = make_perc({})
    perc = make_perc({_t(base, 30): [Detection("cup", 0.9, (190.0, 330.0, 210.0, 350.0))]})
    assert sn.item_proposals(make_scene([True] * 40, [[]] * 40), perc) == []


def test_static_interior_is_not_a_proposal():
    base = make_perc({})
    bag = Detection("handbag", 0.6, (100.0, 380.0, 140.0, 420.0))
    perc = make_perc({t: [bag] for t in base.times})          # «сумка» видна весь ролик в одной точке — интерьер
    assert sn.item_proposals(make_scene([True] * 40, [[]] * 40), perc) == []


def test_item_held_by_visitor_is_ignored():
    base = make_perc({})
    phone = Detection("cell phone", 0.6, (395.0, 300.0, 415.0, 320.0))   # в рамке посетителя, вне зоны стола
    perc = make_perc({_t(base, k): [phone] for k in (30, 31, 33)}, cashier_box=(60, 250, 160, 480), visitor_box=(360, 200, 460, 500))
    assert sn.item_proposals(make_scene([True] * 40, [[2]] * 40), perc) == []


def test_crop_box_is_square_and_inside_frame():
    for box in [(0, 0, 10, 10), (450, 830, 464, 848), (200, 400, 230, 430)]:
        x1, y1, x2, y2 = sn.crop_box(box, 464, 848)
        assert 0 <= x1 < x2 <= 464 and 0 <= y1 < y2 <= 848
        assert abs((x2 - x1) - (y2 - y1)) <= 1
        assert (x2 - x1) >= int(sn.CROP_MIN_FRAC * 464)


# ----------------------------------------------------------------------------- проверка VLM


def make_asker(answer, budget=30.0):
    cache = sn.AnswerCache("/nonexistent.mov")
    calls = []

    class V:
        def ask_yes_no(self, image, question):
            t = float(image[0, 0, 0]) / 10.0      # время зашито в кадр
            calls.append((t, question))
            return answer(t, question)

    def read_frame(t):
        fr = np.zeros((848, 464, 3), np.uint8)
        fr[:, :, :] = int(round(t * 10))
        return fr

    asker = sn._Asker(V(), cache, budget, read_frame)
    return asker, calls


def phone_case():
    """Телефон у кассира 6–9 с; ролик 20 с."""
    base = make_perc({})
    ks = [k for k, t in enumerate(base.times) if 6.0 <= t <= 9.0]
    phone = Detection("cell phone", 0.5, (190.0, 330.0, 210.0, 350.0))
    perc = make_perc({_t(base, k): [phone] for k in ks[::2]})
    props = sn.item_proposals(make_scene([True] * 40, [[]] * 40), perc)
    assert len(props) == 1
    return props[0], perc


def _main_q(q):
    return q.startswith(sn.PROBES["phone"].question)


def test_verified_when_model_confirms_and_control_is_clear():
    p, perc = phone_case()
    asker, calls = make_asker(lambda t, q: (6.0 <= t <= 9.5) if _main_q(q) else False)
    rep = sn.verify_proposal(p, asker, perc, 20.0)
    assert rep["ok"] is True, rep
    assert len(rep["yes"]) >= 2 and rep["control"]["answer"] is False
    assert not (6.0 - sn.CONTROL_CLEAR_SEC <= rep["control"]["t"] <= 9.0 + sn.CONTROL_CLEAR_SEC)
    ev = sn._proposal_event(p, rep)
    assert ev.type_code == "T8" and ev.extras["source"] == sn.SOURCE and ev.type_code in RULES
    assert abs(ev.timecode_sec - p.t0) < 1e-6 and 1 <= len(ev.evidence_times) <= 3
    assert "требует проверки" in ev.description


def test_rejected_when_model_says_yes_everywhere():
    """Модель «всегда да» на этот участок кадра (смещение к «да») — контроль ловит."""
    p, perc = phone_case()
    asker, _ = make_asker(lambda t, q: True if _main_q(q) else False)
    rep = sn.verify_proposal(p, asker, perc, 20.0)
    assert rep["ok"] is False and rep["reason"] == "control_failed"


def test_rejected_when_refuted():
    """«Это купюры?» → да: телефон не подтверждён (купюры в руках кассира — штатная работа)."""
    p, perc = phone_case()
    asker, _ = make_asker(lambda t, q: (6.0 <= t <= 9.5))
    rep = sn.verify_proposal(p, asker, perc, 20.0)
    assert rep["ok"] is False and rep["reason"] == "vlm_not_confirmed" and rep.get("refuted_at")


@pytest.mark.parametrize("answer", [None, False])
def test_rejected_on_unknown_or_no(answer):
    p, perc = phone_case()
    asker, _ = make_asker(lambda t, q: answer if _main_q(q) else False)
    assert sn.verify_proposal(p, asker, perc, 20.0)["ok"] is False


def test_rejected_with_single_yes():
    p, perc = phone_case()
    first = {}

    def ans(t, q):
        if not _main_q(q):
            return False
        if "t" not in first and 6.0 <= t <= 9.5:
            first["t"] = t
        return t == first.get("t")

    asker, _ = make_asker(ans)
    assert sn.verify_proposal(p, asker, perc, 20.0)["ok"] is False


def test_rejected_when_item_visible_whole_video():
    base = make_perc({})
    # виден почти всегда, но в разных точках кадра (не интерьер) — контрольного момента нет
    items = {t: [Detection("cell phone", 0.5, (190.0 + 30 * (k % 3), 330.0, 210.0 + 30 * (k % 3), 350.0))] for k, t in enumerate(base.times)}
    perc = make_perc(items)
    props = [p for p in sn.item_proposals(make_scene([True] * 40, [[]] * 40), perc)]
    assert props
    asker, _ = make_asker(lambda t, q: True if _main_q(q) else False)
    for p in props:
        rep = sn.verify_proposal(p, asker, perc, 20.0)
        assert rep["ok"] is False


def test_control_requires_cashier_in_the_crop():
    """Контроль «нет» на пустом участке ничего не доказывает: с ролями контрольный момент — только когда рамка
    кассира заходит во фрагмент. Кассир (трек 1) не числится кассиром → контрольного момента нет → отказ."""
    p, perc = phone_case()
    asker, _ = make_asker(lambda t, q: (6.0 <= t <= 9.5) if _main_q(q) else False)
    rep = sn.verify_proposal(p, asker, perc, 20.0, roles={1: "visitor"})
    assert rep["ok"] is False and rep["reason"] == "no_control_frame"
    asker, _ = make_asker(lambda t, q: (6.0 <= t <= 9.5) if _main_q(q) else False)
    assert sn.verify_proposal(p, asker, perc, 20.0, roles={1: "cashier"})["ok"] is True


def bag_case():
    """Кассир вносит сумку 5–7 с (детектор видит её у кассира), дальше детектор сумку не видит."""
    base = make_perc({})
    ks = [k for k, t in enumerate(base.times) if 5.0 <= t <= 7.0]
    bag = Detection("handbag", 0.7, (190.0, 300.0, 230.0, 360.0))
    perc = make_perc({_t(base, k): [bag] for k in ks})
    props = sn.item_proposals(make_scene([True] * 40, [[]] * 40), perc)
    assert len(props) == 1 and props[0].probe == "bag"
    return props[0], perc


def _desk_q(q):
    return q.startswith(sn.PROBES["bag"].desk_question)


def test_carried_bag_later_on_desk_is_confirmed():
    p, perc = bag_case()
    asker, _ = make_asker(lambda t, q: (t >= 12.0) if _desk_q(q) else False)
    rep = sn.verify_on_desk(p, asker, make_scene([True] * 40, [[]] * 40), 20.0)
    assert rep["ok"] is True, rep
    assert 12.0 <= rep["t_on_desk"] < 12.0 + sn.FOLLOW_STEP_SEC
    assert rep["control"]["t"] < p.t0 and rep["control"]["answer"] is False
    ev = sn._desk_event(p, rep)
    assert ev.type_code == "T8" and ev.extras["source"] == sn.SOURCE and len(ev.evidence_times) <= 3


def test_carried_bag_needs_two_consecutive_yes_and_clean_control():
    p, perc = bag_case()
    sc = make_scene([True] * 40, [[]] * 40)
    first = {}

    def one_yes(t, q):
        if not _desk_q(q) or t < 12.0:
            return False
        first.setdefault("t", t)
        return t == first["t"]

    asker, _ = make_asker(one_yes)
    assert sn.verify_on_desk(p, asker, sc, 20.0)["ok"] is False
    asker, _ = make_asker(lambda t, q: True if _desk_q(q) else False)       # «сумка на столе» всегда, и до внесения
    rep = sn.verify_on_desk(p, asker, sc, 20.0)
    assert rep["ok"] is False and rep["reason"] == "control_failed"
    asker, _ = make_asker(lambda t, q: (t >= 12.0))                        # да и на «это инкассаторская сумка?»
    assert sn.verify_on_desk(p, asker, sc, 20.0)["ok"] is False


# ----------------------------------------------------------------------------- точка входа


def _ctx(scene):
    ctx = VideoContext(path="/nonexistent.mov", info=INFO, bright_t=np.zeros(0), bright_v=np.zeros(0))
    ctx.scene = scene
    return ctx


def test_disabled_flag_returns_nothing(monkeypatch):
    monkeypatch.setenv("SAFETY_NET", "0")
    ci = [True] * 10 + [False] * 10
    added, rep = sn.run_safety_net(_ctx(make_scene(ci, [[2]] * 20)), [], None)
    assert added == [] and rep.enabled is False


def test_without_vlm_only_structural_layer(monkeypatch):
    monkeypatch.delenv("SAFETY_NET", raising=False)
    monkeypatch.delenv("FEATURES_CACHE", raising=False)
    monkeypatch.setenv("VLM_ENABLED", "0")
    monkeypatch.setattr(vlm, "_describer", None)
    p, perc = phone_case()
    ci = [True] * 10 + [False] * 10
    added, rep = sn.run_safety_net(_ctx(make_scene(ci, [[2]] * 20)), [], perc)
    assert [e.type_code for e in added] == ["T4"]
    # с найденным нарушением предметный слой не запускается (по умолчанию)
    assert rep.vlm == "off" and rep.vlm_calls == 0
    added, rep = sn.run_safety_net(_ctx(make_scene([True] * 40, [[]] * 40)), [], perc)
    assert added == [] and rep.vlm == "unavailable" and rep.proposals == 1


def test_net_never_raises(monkeypatch):
    monkeypatch.setattr(sn, "structural_events", lambda *a: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(sn, "item_proposals", lambda *a: (_ for _ in ()).throw(RuntimeError("boom")))
    added, rep = sn.run_safety_net(_ctx(make_scene([True] * 40, [[]] * 40)), [], make_perc({}))
    assert added == [] and "error" in rep.reason


def test_probe_types_have_legal_grounding():
    for p in sn.PROBES.values():
        assert p.type_code in RULES and p.question.endswith("?")
        assert RULES[p.type_code]["rule"]["url"].startswith("https://adilet.zan.kz/")


# ----------------------------------------------------------------------------- бэкенд llama.cpp


@pytest.fixture
def llama_env(monkeypatch, tmp_path):
    monkeypatch.setenv("VLM_ENABLED", "1")
    monkeypatch.setenv("VLM_BACKEND", "llamacpp")
    monkeypatch.setenv("VLM_GGUF_DIR", str(tmp_path / "gguf"))
    monkeypatch.setenv("VLM_SERVER_URL", "http://127.0.0.1:9")   # заведомо закрытый порт
    monkeypatch.setenv("VLM_LLAMA_BIN", str(tmp_path / "no-such-binary"))
    monkeypatch.setattr(vlm, "_describer", None)
    yield tmp_path
    monkeypatch.setattr(vlm, "_describer", None)


def test_llamacpp_unavailable_without_server_and_weights(llama_env):
    d = vlm.Describer()
    assert d.backend == "llamacpp" and d.available() is False
    assert "GGUF" in d.unavailable_reason()
    assert d.ask_yes_no(np.zeros((32, 32, 3), np.uint8), "Is it?") is None


def test_llamacpp_ask_yes_no_and_latency_gate(llama_env, monkeypatch):
    sent = []

    def fake_http(url, body=None, timeout=5.0):
        if url.endswith("/health"):
            return {"status": "ok"}
        sent.append(body)
        return {"choices": [{"message": {"content": "Yes."}}]}

    monkeypatch.setattr(vlm, "_http_json", fake_http)
    d = vlm.Describer()
    assert d.available() and d.load() and d.loaded
    assert d.ask_yes_no(np.zeros((600, 300, 3), np.uint8), "Is there a phone?") is True
    msg = sent[-1]["messages"][-1]["content"]
    assert msg[0]["type"] == "image_url" and msg[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert sent[-1]["temperature"] == 0
    assert d.fast_enough() is True and d.latency_sec is not None
    monkeypatch.setenv("VLM_PROBE_MAX_SEC", "0")
    d.latency_sec = 0.5
    assert d.fast_enough() is False


def test_llamacpp_garbled_answer_is_none(llama_env, monkeypatch):
    def fake_http(url, body=None, timeout=5.0):
        if url.endswith("/health"):
            return {"status": "ok"}
        return {"choices": [{"message": {"content": "The image shows a desk"}}]}

    monkeypatch.setattr(vlm, "_http_json", fake_http)
    d = vlm.Describer()
    assert d.load()
    assert d.ask_yes_no(np.zeros((64, 64, 3), np.uint8), "Is there a phone?") is None
    assert d.fast_enough() is False, "не распознанный ответ на замере → модель для сетки не используется"


def test_gguf_files_detection(tmp_path):
    d = tmp_path / "g"
    d.mkdir()
    assert vlm.gguf_files(d) is None
    (d / "Model-Q4_K_M.gguf").write_bytes(b"")
    assert vlm.gguf_files(d) is None
    (d / "mmproj-Model-Q8_0.gguf").write_bytes(b"")
    lm, mm = vlm.gguf_files(d)
    assert lm.name.startswith("Model") and mm.name.startswith("mmproj")


def test_llamacpp_backend_does_not_rewrite_descriptions_by_default(llama_env, monkeypatch):
    calls = []

    def fake_http(url, body=None, timeout=5.0):
        if url.endswith("/health"):
            return {"status": "ok"}
        calls.append(body)
        return {"choices": [{"message": {"content": "Кассир находится у стола, на столе лежит сумка и документы."}}]}

    monkeypatch.setattr(vlm, "_http_json", fake_http)
    vs = [{"type_code": "T8", "timecode_sec": 1.0, "description": "шаблонное описание события T8 для теста",
           "evidence_frames": [{"timestamp_sec": 1.0, "image_base64": ""}], "source": "safety_net"}]
    before = [dict(v) for v in vs]
    vlm.describe_violations("/nonexistent.mov", vs, None)
    assert vs == before and calls == []
    monkeypatch.setenv("VLM_DESCRIBE", "1")
    vlm.describe_violations("/nonexistent.mov", vs, None)
    assert vs == before, "описание страховочной сетки не переписывается никогда"
