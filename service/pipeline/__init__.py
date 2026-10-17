"""Пайплайн анализа видео кассы. Публичный интерфейс:

    analyze_video(path, progress=None) -> dict   # {"verdict", "violations", "meta"}
    warmup()                                      # загрузка моделей (вызывается один раз в воркере)

Формат результата — строго по разделу 10 ТЗ + расширения rule/confidence/type_code (ПЛАН 5.6).
"""
from __future__ import annotations

import logging
import time
from typing import Callable, Optional

from .context import Event, VideoContext
from .detectors import DETECTORS
from .draw import caption
from .rules import RULES, api_rule
from .video import brightness_series, encode_jpeg_base64, read_frame_at

PIPELINE_VERSION = "0.1.0-skeleton"
MIN_CONFIDENCE = 0.5          # политика вердикта (ПЛАН 5.4): событие учитывается при уверенности ≥ порога
MAX_EVIDENCE_FRAMES = 3       # ТЗ: 1–3 кадра

log = logging.getLogger("pipeline")

ProgressFn = Callable[[str, Optional[int]], None]


def warmup() -> None:
    """Загрузка моделей/весов. Вызывается один раз при старте воркер-процесса. Пока моделей нет — только импорт cv2."""
    import cv2  # noqa: F401

    log.info("pipeline %s warmed up (cv2 %s)", PIPELINE_VERSION, cv2.__version__)


def _noop(stage: str, percent: Optional[int] = None) -> None:
    return None


def build_evidence(ctx: VideoContext, ev: Event) -> list[dict]:
    times = ev.evidence_times[:MAX_EVIDENCE_FRAMES] or [ev.timecode_sec]
    caps = list(ev.evidence_captions) + [""] * len(times)
    rule = RULES[ev.type_code]
    frames: list[dict] = []
    for ts, cap_text in zip(times, caps):
        ts = max(0.0, min(float(ts), max(ctx.info.duration_sec - 0.05, 0.0)))
        frame = read_frame_at(ctx.path, ts, ctx.info)
        if frame is None:
            continue
        lines = [f"{ev.type_code} · {rule['name']}", f"t = {ts:.1f} с" + (f" · {cap_text}" if cap_text else "")]
        frame = caption(frame, lines)
        frames.append({"timestamp_sec": round(ts, 2), "image_base64": encode_jpeg_base64(frame)})
    if not frames:
        # ТЗ требует хотя бы один кадр: берём кадр события без подписи
        frame = read_frame_at(ctx.path, ev.timecode_sec, ctx.info)
        if frame is not None:
            frames.append({"timestamp_sec": round(ev.timecode_sec, 2), "image_base64": encode_jpeg_base64(frame)})
    return frames


def event_to_violation(ctx: VideoContext, ev: Event) -> dict:
    rule = RULES[ev.type_code]
    out = {
        # --- поля ТЗ ---
        "type": rule["name"],
        "timecode_sec": round(float(ev.timecode_sec), 2),
        "description": ev.description,
        "evidence_frames": build_evidence(ctx, ev),
        # --- расширения ---
        "type_code": ev.type_code,
        "confidence": round(float(ev.confidence), 2),
        "rule": api_rule(ev.type_code),
    }
    if ev.end_sec is not None:
        out["end_sec"] = round(float(ev.end_sec), 2)
    return out


def analyze_video(path: str, progress: Optional[ProgressFn] = None) -> dict:
    progress = progress or _noop
    t0 = time.time()

    progress("decoding", 5)
    bt, bv, info = brightness_series(path, target_fps=10.0)
    ctx = VideoContext(path=path, info=info, bright_t=bt, bright_v=bv, progress=progress)
    log.info("decoded %s: %.1fs, %.2f fps, %dx%d", path, info.duration_sec, info.fps, info.width, info.height)

    # Слои восприятия/состояний (детекции, треки, зоны, состояния) подключаются здесь:
    # progress("detection", 30); ctx.detections = ...; ctx.tracks = ...; ctx.states = ...

    progress("rules", 60)
    events: list[Event] = []
    for det in DETECTORS:
        try:
            found = det(ctx)
        except Exception as exc:  # noqa: BLE001 — одно сломанное правило не должно валить весь анализ
            log.exception("detector %s failed: %s", det.__name__, exc)
            continue
        events.extend(found)
    events = [e for e in events if e.confidence >= MIN_CONFIDENCE]
    events.sort(key=lambda e: (e.timecode_sec, e.type_code))

    progress("evidence", 80)
    violations = [event_to_violation(ctx, ev) for ev in events]

    progress("describing", 95)
    # Здесь — VLM-слой: уточнение описаний по кадрам события (ПЛАН 5.1, п. 4). Пока описания шаблонные.

    result = {
        "verdict": "violation" if violations else "no_violation",
        "violations": violations,
        "meta": {
            "duration_sec": round(info.duration_sec, 2),
            "fps": round(info.fps, 3),
            "frames": info.frame_count,
            "resolution": f"{info.width}x{info.height}",
            "pipeline_version": PIPELINE_VERSION,
            "pipeline_time_sec": round(time.time() - t0, 2),
            "rules_checked": [d.__name__ for d in DETECTORS],
        },
    }
    progress("done", 100)
    return result


__all__ = ["analyze_video", "warmup", "PIPELINE_VERSION", "RULES"]
