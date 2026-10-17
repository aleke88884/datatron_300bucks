"""Проверка ответа GET /api/v1/jobs/{id} на соответствие схеме раздела 10 ТЗ (+ наши расширения)."""
from __future__ import annotations

import base64
import re

import cv2
import numpy as np

TZ_VIOLATION_FIELDS = ("type", "timecode_sec", "description", "evidence_frames")
JOB_ID_RE = re.compile(r"^[0-9a-f]{8}$")


def assert_done_result(j: dict) -> None:
    assert JOB_ID_RE.match(j["job_id"]), j["job_id"]
    assert j["status"] == "done"
    assert j["verdict"] in ("no_violation", "violation")
    assert isinstance(j["violations"], list)
    assert (j["verdict"] == "violation") == (len(j["violations"]) > 0), "вердикт не согласован со списком нарушений"
    # в ответе done не должно быть служебных полей других состояний
    assert "error" not in j and "progress" not in j
    for v in j["violations"]:
        assert_violation(v)


def assert_violation(v: dict) -> None:
    for f in TZ_VIOLATION_FIELDS:
        assert f in v, f"нет поля ТЗ {f}"
    assert isinstance(v["type"], str) and v["type"].strip()
    assert isinstance(v["timecode_sec"], (int, float)) and not isinstance(v["timecode_sec"], bool)
    assert v["timecode_sec"] >= 0
    assert isinstance(v["description"], str) and len(v["description"]) >= 20
    frames = v["evidence_frames"]
    assert 1 <= len(frames) <= 3, "ТЗ: 1–3 кадра-доказательства"
    for fr in frames:
        assert isinstance(fr["timestamp_sec"], (int, float)) and fr["timestamp_sec"] >= 0
        assert_jpeg_base64(fr["image_base64"])
    # расширения
    assert re.match(r"^T\d{1,2}$", v["type_code"])
    assert 0.0 <= v["confidence"] <= 1.0
    rule = v["rule"]
    assert rule["document"] and rule["clause"] and rule["url"].startswith("https://adilet.zan.kz/rus/docs/")


def assert_jpeg_base64(s: str) -> np.ndarray:
    assert isinstance(s, str) and not s.startswith("data:"), "image_base64 — чистый base64 без префикса data:"
    raw = base64.b64decode(s, validate=True)
    assert raw[:3] == b"\xff\xd8\xff", "не JPEG"
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    assert img is not None and img.shape[0] > 0 and img.shape[1] > 0, "JPEG не декодируется"
    return img
