"""Пайплайн напрямую на всех видео датасета против разметки open_labels.xlsx.
Проверяет схему результата для каждого видео, время обработки, контекст T9 на Х-11/Х-12, отсутствие ложных
срабатываний на чистых видео и генерацию кадров-доказательств. Полнота (recall) по 25 нарушениям — xfail,
пока правила T1–T8, T10–T13 — заглушки, а T9 переведён в контекстный режим."""
from __future__ import annotations

import re
import sys
import time

import numpy as np
import pytest

from .conftest import LABELS_XLSX, SERVICE_DIR, VIDEOS_DIR
from .schema import assert_jpeg_base64, assert_violation

pytestmark = pytest.mark.dataset

if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

LIGHTS_OFF_VIDEOS = {"Х-11", "Х-12"}   # в обоих кассир выключает свет; нарушение только в Х-11 (экран активен, сейф открыт)
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


def test_metrics_report(results, labels):
    """Таблица метрик (ПЛАН 8.2). Информативна; пороги мягкие, пока правила — заглушки (см. xfail ниже)."""
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


@pytest.mark.xfail(reason="реализован только контекстный T9; правила T1–T8, T10–T13 — заглушки", strict=True)
def test_full_recall(results, labels):
    missed = [n for n in results if labels[n]["violation"] and results[n]["verdict"] != "violation"]
    assert missed == []
