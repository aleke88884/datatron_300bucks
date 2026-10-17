"""Подменный пайплайн для тестов устойчивости воркера и прохождения нарушений через API.
Поведение задаётся первыми байтами файла: SLOW — спит дольше таймаута; CRASH — убивает процесс воркера;
FAIL — бросает исключение; VIOL — возвращает нарушение с настоящим JPEG; иначе — короткая обработка с прогрессом."""
import base64
import os
import time

PIPELINE_VERSION = "fake-test"


# дорожки состояний сцены в формате pipeline/timeline.py — чтобы UI и API проверялись на непустых данных
TIMELINE = {
    "step": 0.5, "duration": 6.0, "anchors": {"desk": True, "safe": True, "laptop": True},
    "lanes": {"cashier": [[0.0, 1.5]], "at_desk": [[0.0, 1.5]], "visitors": [[3.0, 6.0, 1]], "safe_open": [],
              "safe_unknown": [], "desk_items": [[0.5, 6.0]], "bag": [], "desk_unknown": [], "screen_active": [[0.0, 6.0]],
              "screen_locked": [], "screen_unknown": [], "lights_off": []},
    "unreliable": [],
}


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


def analyze_video(path, progress=None):
    progress = progress or (lambda stage, percent=None: None)
    with open(path, "rb") as f:
        head = f.read(8)
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
            "meta": {"pipeline_version": PIPELINE_VERSION, "duration_sec": 6.0, "timeline": TIMELINE},
        }
    progress("fake-stage", 42)
    time.sleep(1.5)
    return {"verdict": "no_violation", "violations": [], "meta": {"pipeline_version": PIPELINE_VERSION}}
