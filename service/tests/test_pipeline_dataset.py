"""Пайплайн напрямую на всех видео датасета против разметки open_labels.xlsx.
Проверяет схему результата для каждого видео, время обработки, контекст T9 на Х-11/Х-12, нарушение T3 (экран АРМ
активен после выключения света) ровно на Х-11, отсутствие ложных срабатываний на чистых видео и генерацию
кадров-доказательств (T9 — подпись, T3 — рамка вокруг экрана). Полнота (recall) по 22 нарушениям — xfail,
пока правила T1, T2, T4–T8, T10–T13 — заглушки, а T9 — контекст."""
from __future__ import annotations

import re
import sys
import time

import cv2
import numpy as np
import pytest

from .conftest import LABELS_XLSX, SERVICE_DIR, VIDEOS_DIR
from .schema import assert_jpeg_base64, assert_violation

pytestmark = pytest.mark.dataset

if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

LIGHTS_OFF_VIDEOS = {"Х-11", "Х-12"}   # в обоих кассир выключает свет; нарушение только в Х-11 (экран ноутбука активен — T3)
SCREEN_ON_VIDEOS = {"Х-11", "Х-28"}    # где T3 должен сработать: Х-11 (экран светится в темноте), Х-28 (кассир ушёл, экран активен)
# Ожидаемые срабатывания правил по датасету (офлайн-оценка scripts/eval_rules.py, 18.10.2026): 12 из 22 нарушений, 0 ложных
EXPECTED_TP = {"Х-11", "Х-13", "Х-15", "Х-17", "Х-18", "Х-24", "Х-28", "Х-30", "Х-33", "Х-38", "Х-40", "Х-46"}
EXPECTED_TYPES = {"Х-17": "T10", "Х-18": "T4", "Х-30": "T5", "Х-33": "T5", "Х-38": "T7", "Х-40": "T8", "Х-46": "T8", "Х-15": "T1", "Х-24": "T1"}
LABEL_TOLERANCE_SEC = 3.0


def _norm(name: str) -> str:
    """'X-1' (латиница) и 'Х-1' (кириллица) → единый ключ."""
    return name.strip().replace("X", "Х").replace(".mov", "")


def _t9_context(res: dict) -> list[dict]:
    return [e for e in res["meta"].get("context_events") or [] if e["type_code"] == "T9"]


@pytest.fixture(scope="module")
def labels() -> dict[str, dict]:
    if not LABELS_XLSX.exists():
        pytest.skip("нет open/open_labels.xlsx")
    import openpyxl

    ws = openpyxl.load_workbook(LABELS_XLSX, read_only=True).active
    out = {}
    for row in list(ws.iter_rows(values_only=True))[1:]:
        if not row or not row[0]:
            continue
        name, ts, verdict = row[0], row[1], row[2]
        m = re.search(r"\d+(?:[.,]\d+)?", str(ts or ""))
        out[_norm(str(name))] = {
            "violation": "без" not in str(verdict).lower(),
            "timecode": float(m.group().replace(",", ".")) if m else None,
        }
    assert len(out) >= 40
    return out


@pytest.fixture(scope="module")
def results() -> dict[str, dict]:
    if not VIDEOS_DIR.exists():
        pytest.skip("нет open/videos")
    from pipeline import analyze_video

    out = {}
    for p in sorted(VIDEOS_DIR.glob("*.mov")):
        t0 = time.time()
        res = analyze_video(str(p))
        res["_elapsed"] = time.time() - t0
        out[_norm(p.name)] = res
    assert len(out) >= 40
    return out


def test_labels_cover_all_videos(labels, results):
    assert set(results) == set(labels), set(results) ^ set(labels)


def test_every_result_matches_schema(results):
    for name, res in results.items():
        assert res["verdict"] in ("no_violation", "violation"), name
        assert (res["verdict"] == "violation") == bool(res["violations"]), name
        assert isinstance(res["meta"]["context_events"], list), name
        for v in res["violations"]:
            assert_violation(v)
            assert v["timecode_sec"] <= res["meta"]["duration_sec"], name
            for f in v["evidence_frames"]:
                assert f["timestamp_sec"] <= res["meta"]["duration_sec"] + 0.01, name
        for e in res["meta"]["context_events"]:
            assert set(e) >= {"type_code", "name", "timecode_sec", "end_sec", "description"}, name
            assert 0 <= e["timecode_sec"] <= res["meta"]["duration_sec"], name


def test_durations_match_dataset_description(results):
    for name, res in results.items():
        d = res["meta"]["duration_sec"]
        assert 10 <= d <= 145, f"{name}: {d}"
        assert res["meta"]["resolution"] == "464x848", name
        assert 29 <= res["meta"]["fps"] <= 31, name


def test_processing_time_within_limit(results):
    slowest = max(results.items(), key=lambda kv: kv[1]["_elapsed"])
    total = sum(r["_elapsed"] for r in results.values())
    print(f"\nсамое долгое: {slowest[0]} {slowest[1]['_elapsed']:.1f} с; всего {total:.1f} с на {len(results)} видео")
    for name, res in results.items():
        assert res["_elapsed"] < 240, f"{name}: {res['_elapsed']:.1f} с"


def test_t9_context_on_x11_near_label(results, labels):
    res = results["Х-11"]
    t9 = _t9_context(res)
    assert len(t9) == 1, "ровно одно контекстное событие T9"
    assert abs(t9[0]["timecode_sec"] - labels["Х-11"]["timecode"]) <= LABEL_TOLERANCE_SEC
    assert t9[0]["end_sec"] == pytest.approx(res["meta"]["duration_sec"], abs=0.2), "свет не включается до конца ролика"
    assert all(v["type_code"] != "T9" for v in res["violations"]), "T9 — контекст, не нарушение"


def test_t9_context_fires_only_where_lights_go_off(results):
    fired = {n for n, r in results.items() if _t9_context(r)}
    assert fired == LIGHTS_OFF_VIDEOS, fired


def _t3(res: dict) -> list[dict]:
    return [v for v in res["violations"] if v["type_code"] == "T3"]


def test_t3_violation_on_x11_near_label(results, labels):
    """Х-11 по разметке — нарушение на 18 с: вердикт violation, ровно одно событие T3 с таймкодом в пределах
    3 с от разметки, экран активен до конца ролика, 3 кадра-доказательства, ссылка на п. 14 Правил № 231."""
    res = results["Х-11"]
    assert res["verdict"] == "violation"
    assert [v["type_code"] for v in res["violations"]] == ["T3"]
    t3 = _t3(res)[0]
    assert abs(t3["timecode_sec"] - labels["Х-11"]["timecode"]) <= LABEL_TOLERANCE_SEC
    assert t3["timecode_sec"] == pytest.approx(_t9_context(res)[0]["timecode_sec"], abs=0.01), "таймкод = момент выключения света"
    assert t3["end_sec"] == pytest.approx(res["meta"]["duration_sec"], abs=0.2)
    assert t3["confidence"] >= 0.7
    assert t3["rule"]["clause"] == "п. 14" and t3["rule"]["url"].endswith("V1900019680") and t3["rule"].get("note")
    assert "экран" in t3["description"] and "освещение" in t3["description"]
    assert not any(w in t3["description"].lower() for w in ("похит", "украл", "кража")), "только наблюдаемое"
    ts = [f["timestamp_sec"] for f in t3["evidence_frames"]]
    assert len(ts) == 3 and ts == sorted(ts) and ts[0] < t3["timecode_sec"] <= ts[1] < ts[2]


def test_t3_fires_only_where_screen_active(results):
    fired = {n for n, r in results.items() if _t3(r)}
    assert fired == SCREEN_ON_VIDEOS, fired
    assert results["Х-12"]["verdict"] == "no_violation", "в Х-12 свет тоже гаснет, но экрана нет — нарушения нет"


def test_expected_true_positives(results, labels):
    """Правила находят зафиксированный набор нарушений (регресс-ориентир офлайн-оценки)."""
    found = {n for n, r in results.items() if r["verdict"] == "violation" and labels[n]["violation"]}
    missing = EXPECTED_TP - found
    assert not missing, f"пропали ожидаемые срабатывания: {sorted(missing)}"


def test_expected_types(results):
    for name, code in EXPECTED_TYPES.items():
        codes = [v["type_code"] for v in results[name]["violations"]]
        assert code in codes, f"{name}: ожидали {code}, получили {codes}"


def test_timecodes_near_labels(results, labels):
    """Для найденных нарушений хотя бы одно событие в пределах 3 с от разметки — не менее чем в половине случаев."""
    ok = tot = 0
    for n, r in results.items():
        if r["verdict"] != "violation" or not labels[n]["violation"] or labels[n]["timecode"] is None:
            continue
        tot += 1
        if min(abs(v["timecode_sec"] - labels[n]["timecode"]) for v in r["violations"]) <= LABEL_TOLERANCE_SEC:
            ok += 1
    assert tot == 0 or ok / tot >= 0.5, f"в допуске {ok} из {tot}"


def test_no_false_positives_on_clean_videos(results, labels):
    fps = sorted(n for n, r in results.items() if r["verdict"] == "violation" and not labels[n]["violation"])
    assert fps == [], f"ложные срабатывания на чистых видео: {fps}"


def test_t9_event_to_violation_produces_real_evidence():
    """Генерация кадров-доказательств настоящим пайплайном (T9 на Х-11): 3 JPEG исходного размера,
    кадр до события светлый, кадр события тёмный, сверху нарисована подпись, остальная часть кадра не изменена."""
    from pipeline import event_to_violation
    from pipeline.context import VideoContext
    from pipeline.detectors import detect_t9_lights_off
    from pipeline.video import brightness_series, read_frame_at

    path = VIDEOS_DIR / "Х-11.mov"
    if not path.exists():
        pytest.skip("нет Х-11.mov")
    bt, bv, info = brightness_series(str(path))
    ctx = VideoContext(path=str(path), info=info, bright_t=bt, bright_v=bv)
    events = detect_t9_lights_off(ctx)
    assert len(events) == 1 and events[0].type_code == "T9"
    ev = events[0]
    assert ev.timecode_sec == pytest.approx(16.06, abs=0.15)
    assert ev.confidence >= 0.9

    v = event_to_violation(ctx, ev)
    assert_violation(v)
    assert v["type_code"] == "T9" and v["rule"]["clause"] == "п. 16 пп. 6)"
    ts = [f["timestamp_sec"] for f in v["evidence_frames"]]
    assert len(ts) == 3 and ts == sorted(ts) and ts[1] == pytest.approx(ev.timecode_sec, abs=0.01)
    before, at = (assert_jpeg_base64(f["image_base64"]) for f in v["evidence_frames"][:2])
    assert before.shape == at.shape == (848, 464, 3)
    raw_before = read_frame_at(str(path), ts[0], info)
    raw_at = read_frame_at(str(path), ts[1], info)
    assert raw_before.mean() > 100 and raw_at.mean() < 40, "до события светло, в момент события темно"
    # подпись: верхняя полоса отличается от исходника, нижняя половина совпадает (с точностью до JPEG)
    top_diff = np.abs(at[:40].astype(int) - raw_at[:40].astype(int)).mean()
    bottom_diff = np.abs(at[424:].astype(int) - raw_at[424:].astype(int)).mean()
    assert top_diff > 15 and bottom_diff < 4, (top_diff, bottom_diff)


def test_t3_evidence_frames_have_box_around_screen():
    """T3 на Х-11 напрямую: контекст T9 → событие T3 с областью экрана; кадр «свет выключен» тёмный, вокруг экрана
    нарисована красная рамка (draw.box), сверху подпись. На Х-12 (свет гаснет, ноутбук закрыт) — событий нет."""
    from pipeline import event_to_violation
    from pipeline.context import VideoContext
    from pipeline.detectors import T3_MIN_CONTRAST, detect_t3_workstation_unlocked, detect_t9_lights_off
    from pipeline.video import brightness_series, read_frame_at

    def run(name):
        path = VIDEOS_DIR / f"{name}.mov"
        if not path.exists():
            pytest.skip(f"нет {path.name}")
        bt, bv, info = brightness_series(str(path))
        ctx = VideoContext(path=str(path), info=info, bright_t=bt, bright_v=bv)
        assert detect_t3_workstation_unlocked(ctx) == [], "без контекста T9 правило молчит"
        ctx.context_events = detect_t9_lights_off(ctx)
        assert len(ctx.context_events) == 1
        return ctx, detect_t3_workstation_unlocked(ctx)

    _ctx12, events12 = run("Х-12")
    assert events12 == []

    ctx, events = run("Х-11")
    assert len(events) == 1 and events[0].type_code == "T3"
    ev = events[0]
    assert ev.timecode_sec == pytest.approx(ctx.context_events[0].timecode_sec, abs=0.01)
    reg = ev.extras["region"]
    x1, y1, x2, y2 = reg["xyxy"]
    assert 240 <= x1 < x2 <= 300 and 340 <= y1 < y2 <= 415, "экран ноутбука — справа на столе"
    assert reg["contrast"] >= T3_MIN_CONTRAST and reg["mean"] >= 170 and reg["stability"] >= 0.8
    assert ev.extras["evidence_boxes"] == [reg["xyxy"]] * 3

    v = event_to_violation(ctx, ev)
    assert_violation(v)
    assert v["type_code"] == "T3" and v["rule"]["clause"] == "п. 14"
    ts = [f["timestamp_sec"] for f in v["evidence_frames"]]
    assert len(ts) == 3
    dark = assert_jpeg_base64(v["evidence_frames"][1]["image_base64"])
    raw = read_frame_at(str(ctx.path), ts[1], ctx.info)
    assert dark.shape == raw.shape == (848, 464, 3)
    assert raw.mean() < 40, "кадр события — тёмный (свет выключен)"
    box_gray = cv2.cvtColor(raw[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    assert (box_gray > 150).mean() >= 0.3, "а экран внутри рамки светится (экран занимает ~половину bbox, остальное — рамка ноутбука)"
    # рамка: пиксели по контуру bbox красные (BGR ≈ 60,60,230), внутри экрана кадр не изменён (с точностью до JPEG)
    border = np.concatenate([dark[y1, x1:x2], dark[y2, x1:x2], dark[y1:y2, x1], dark[y1:y2, x2]]).astype(int)
    assert border[:, 2].mean() > 150 and border[:, 0].mean() < 130, border.mean(axis=0)
    inner = slice(y1 + 4, y2 - 4), slice(x1 + 4, x2 - 4)
    assert np.abs(dark[inner].astype(int) - raw[inner].astype(int)).mean() < 8
    # нижняя часть кадра (без подписи и рамки) совпадает с исходником
    assert np.abs(dark[500:].astype(int) - raw[500:].astype(int)).mean() < 4


def test_context_phase_runs_before_violation_rules(monkeypatch):
    """Контекстные правила (T9) выполняются первой фазой: правило-нарушение видит ctx.context_events."""
    import pipeline
    from pipeline.detectors import detect_t9_lights_off

    path = VIDEOS_DIR / "Х-12.mov"
    if not path.exists():
        pytest.skip("нет Х-12.mov")
    seen: list = []

    def detect_t1_valuables_unattended(ctx):  # имя как у настоящего правила: код берётся из имени
        seen.append([e.type_code for e in ctx.context_events])
        return []

    monkeypatch.setattr(pipeline, "DETECTORS", [detect_t1_valuables_unattended, detect_t9_lights_off])
    res = pipeline.analyze_video(str(path))
    assert seen == [["T9"]], seen
    assert res["verdict"] == "no_violation" and [e["type_code"] for e in res["meta"]["context_events"]] == ["T9"]
    assert res["meta"]["rules_checked"] == ["detect_t1_valuables_unattended", "detect_t9_lights_off"]


def test_metrics_report(results, labels):
    """Таблица метрик (ПЛАН 8.2). Информативна; пороги мягкие, пока большинство правил — заглушки (см. xfail ниже)."""
    tp = [n for n in results if results[n]["verdict"] == "violation" and labels[n]["violation"]]
    fn = [n for n in results if results[n]["verdict"] == "no_violation" and labels[n]["violation"]]
    fp = [n for n in results if results[n]["verdict"] == "violation" and not labels[n]["violation"]]
    tn = [n for n in results if results[n]["verdict"] == "no_violation" and not labels[n]["violation"]]
    errs = []
    for n in tp:
        lbl = labels[n]["timecode"]
        if lbl is not None:
            errs.append(min(abs(v["timecode_sec"] - lbl) for v in results[n]["violations"]))
    print("\n=== метрики по 42 видео (вердикт на всё видео) ===")
    print(f"TP={len(tp)} FN={len(fn)} FP={len(fp)} TN={len(tn)}; recall={len(tp)/max(1, len(tp)+len(fn)):.2f}; FP на чистых={len(fp)}/{len(fp)+len(tn)}")
    if errs:
        print(f"ошибка таймкода по найденным: средняя {sum(errs)/len(errs):.2f} с, макс {max(errs):.2f} с")
    print("пропущены:", ", ".join(sorted(fn)))
    clean = [n for n in labels if not labels[n]["violation"]]
    print(f"по разметке: нарушений {len(labels) - len(clean)}, чистых {len(clean)} (в ПЛАН 2 указано 25/17 — неверно)")
    assert len(fp) == 0, "ложные срабатывания на чистых видео недопустимы (FAQ п. 16)"
    assert len(tn) == len(clean)


@pytest.mark.xfail(reason="правила находят 12 из 22 нарушений (см. EXPECTED_TP); T2/T6/T13 выключены, T11/T12 не реализованы", strict=True)
def test_full_recall(results, labels):
    missed = [n for n in results if labels[n]["violation"] and results[n]["verdict"] != "violation"]
    assert missed == []
