"""Подменный пайплайн для тестов устойчивости воркера и прохождения нарушений через API.
Поведение задаётся первыми байтами файла: SLOW — спит дольше таймаута; CRASH — убивает процесс воркера;
FAIL — бросает исключение; VIOL — возвращает нарушение с настоящим JPEG; иначе — короткая обработка с прогрессом.
Для проверки деградации: PART — отдаёт partial() и зависает (жёсткий дедлайн → done с частичным результатом);
LCRASH — падает, если не light-режим (повтор в лёгком режиме успешен); DECO — бросает VideoDecodeError;
DEAD — возвращает в meta, за сколько секунд до мягкого/жёсткого дедлайна пришло задание; SPIN — быстрый ответ без пауз."""
import base64
import os
import time

PIPELINE_VERSION = "fake-test"


def warmup() -> None:
    return None


def _jpeg() -> str:
    import cv2
    import numpy as np

    img = np.full((64, 48, 3), 90, np.uint8)
    cv2.rectangle(img, (8, 8), (40, 56), (60, 60, 230), 2)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return base64.b64encode(buf.tobytes()).decode("ascii")


class VideoDecodeError(RuntimeError):
    pass


def _viol_result(meta=None):
    return {
        "verdict": "violation",
        "violations": [
            {
                "type": "Тестовое нарушение (подменный пайплайн)",
                "timecode_sec": 1.5,
                "description": "Синтетическое событие для проверки прохождения нарушения через API без изменений.",
                "evidence_frames": [{"timestamp_sec": 1.5, "image_base64": _jpeg()}],
                "type_code": "T1",
                "confidence": 0.91,
                "rule": {"document": "Правила № 231 (тест)", "clause": "п. 20", "url": "https://adilet.zan.kz/rus/docs/V1900019680"},
            }
        ],
        "meta": meta or {"pipeline_version": PIPELINE_VERSION},
    }


def analyze_video(path, progress=None, deadline_ts=None, hard_deadline_ts=None, partial=None, light=False):
    progress = progress or (lambda stage, percent=None: None)
    with open(path, "rb") as f:
        head = f.read(8)
    if head.startswith(b"PART"):
        partial(_viol_result())
        time.sleep(120)
    if head.startswith(b"LCRASH"):
        if not light:
            os._exit(1)
        return {"verdict": "no_violation", "violations": [], "meta": {"pipeline_version": PIPELINE_VERSION, "degraded": True, "light": True}}
    if head.startswith(b"DECO"):
        raise VideoDecodeError("cannot open video: fake")
    if head.startswith(b"DEAD"):
        now = time.time()
        return {"verdict": "no_violation", "violations": [], "meta": {
            "pipeline_version": PIPELINE_VERSION, "soft_in": None if deadline_ts is None else deadline_ts - now,
            "hard_in": None if hard_deadline_ts is None else hard_deadline_ts - now, "pid": os.getpid()}}
    if head.startswith(b"SPIN"):
        time.sleep(1.5)
        return {"verdict": "no_violation", "violations": [], "meta": {"pipeline_version": PIPELINE_VERSION, "pid": os.getpid()}}
    if head.startswith(b"SLOW"):
        progress("sleeping", 10)
        time.sleep(120)
    if head.startswith(b"CRASH"):
        os._exit(1)
    if head.startswith(b"FAIL"):
        raise RuntimeError("boom")
    if head.startswith(b"VIOL"):
        return {
            "verdict": "violation",
            "violations": [
                {
                    "type": "Тестовое нарушение (подменный пайплайн)",
                    "timecode_sec": 1.5,
                    "description": "Синтетическое событие для проверки прохождения нарушения через API без изменений.",
                    "evidence_frames": [{"timestamp_sec": 1.5, "image_base64": _jpeg()}, {"timestamp_sec": 2.5, "image_base64": _jpeg()}],
                    "type_code": "T1",
                    "confidence": 0.91,
                    "rule": {"document": "Правила № 231 (тест)", "clause": "п. 20", "url": "https://adilet.zan.kz/rus/docs/V1900019680"},
                    "end_sec": 4.0,
                }
            ],
            "meta": {"pipeline_version": PIPELINE_VERSION},
        }
    progress("fake-stage", 42)
    time.sleep(1.5)
    return {"verdict": "no_violation", "violations": [], "meta": {"pipeline_version": PIPELINE_VERSION}}
