"""Пайплайн анализа видео кассы. Публичный интерфейс:

    analyze_video(path, progress=None) -> dict   # {"verdict", "violations", "meta"}
    warmup()                                      # загрузка моделей (вызывается один раз в воркере)

Формат результата — строго по разделу 10 ТЗ + расширения rule/confidence/type_code (ПЛАН 5.6).
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Callable, Optional

from . import budget as _budget
from .context import Event, VideoContext
from .detectors import DETECTORS
from .draw import box as draw_box
from .draw import caption
from .rules import RULES, api_rule
from .video import brightness_series, encode_jpeg_base64, read_frame_at

PIPELINE_VERSION = "0.3.0"
MIN_CONFIDENCE = 0.5          # политика вердикта (ПЛАН 5.4): событие учитывается при уверенности ≥ порога
MAX_EVIDENCE_FRAMES = 3       # ТЗ: 1–3 кадра
# Контекстные события: описывают условия сцены, но САМИ ПО СЕБЕ не являются нарушением и не дают вердикт.
# T9 (свет выключен) встречается и в чистом Х-12, и в Х-11; нарушение в Х-11 — то, что кассир уходит в темноте,
# оставив экран активным (T3; реализовано) / сейф открытым (T2). Контекстные правила запускаются первой фазой,
# их события лежат в ctx.context_events до запуска остальных правил и попадают в meta.context_events ответа.
CONTEXT_ONLY_TYPES = {"T9"}

log = logging.getLogger("pipeline")

ProgressFn = Callable[[str, Optional[int]], None]


def warmup() -> None:
    """Загрузка моделей/весов. Вызывается один раз при старте воркер-процесса. Пока моделей нет — только импорт cv2."""
    import cv2  # noqa: F401

    try:
        from .detect import get_detector

        det = get_detector()
        # холостой прогон: инициализация потоков/памяти ORT не должна попасть в время первого задания
        import numpy as np

        det(np.zeros((848, 464, 3), np.uint8))
    except Exception as exc:  # noqa: BLE001
        log.warning("detector not available at warmup: %s", exc)
    try:
        from .vlm import get_describer

        d = get_describer()
        if d.available():
            d.load()
    except Exception as exc:  # noqa: BLE001
        log.warning("vlm not available at warmup: %s", exc)
    log.info("pipeline %s warmed up (cv2 %s)", PIPELINE_VERSION, cv2.__version__)


def _noop(stage: str, percent: Optional[int] = None) -> None:
    return None


def _features_cache_dir() -> Optional[Path]:
    """Папка с заранее посчитанными признаками (results/): perception/<видео>.json и states/<видео>.json.
    Используется офлайн-оценкой правил и тестами; в проде не задаётся — всё считается по видео."""
    p = os.environ.get("FEATURES_CACHE")
    return Path(p) if p else None


def _perceive(path: str, info, progress):
    """Perception + states: из кеша признаков (если задан FEATURES_CACHE и файлы есть) или по видео."""
    from .perception import cached_perception, perception_from_dict
    from .states import extract_states

    stem = Path(path).stem.replace("X", "Х")
    cache = _features_cache_dir()
    perc = states = None
    if cache is not None:
        pj = cache / "perception" / f"{stem}.json"
        sj = cache / "states" / f"{stem}.json"
        if pj.exists():
            perc = perception_from_dict(json.loads(pj.read_text(encoding="utf-8")), info)
            log.info("perception from cache %s", pj)
        if sj.exists() and perc is not None:
            states = json.loads(sj.read_text(encoding="utf-8"))
            if (states.get("scene") or {}).get("version") != 2:
                states = None
    if perc is None:
        progress("detection", 20)
        perc = cached_perception(path, info, progress=progress)
    if states is None:
        progress("states", 45)
        states = extract_states(path, info, perc, progress=progress)
        if cache is not None and not (states.get("scene") or {}).get("degrade"):
            (cache / "states").mkdir(parents=True, exist_ok=True)
            (cache / "states" / f"{stem}.json").write_text(json.dumps(states, ensure_ascii=False), encoding="utf-8")
    return perc, states


def detector_code(det: Callable) -> str:
    """detect_t9_lights_off -> 'T9' (по имени функции; порядок/коды проверяет tests/test_rules.py)."""
    m = re.search(r"_t(\d{1,2})_", det.__name__)
    return f"T{m.group(1)}" if m else ""


def _run_detectors(ctx: VideoContext, dets: list[Callable]) -> list[Event]:
    events: list[Event] = []
    for det in dets:
        try:
            found = det(ctx)
        except Exception as exc:  # noqa: BLE001 — одно сломанное правило не должно валить весь анализ
            log.exception("detector %s failed: %s", det.__name__, exc)
            continue
        events.extend(found)
    events = [e for e in events if e.confidence >= MIN_CONFIDENCE]
    events.sort(key=lambda e: (e.timecode_sec, e.type_code))
    return events


def build_evidence(ctx: VideoContext, ev: Event) -> list[dict]:
    if os.environ.get("EVIDENCE_FRAMES") == "0":
        return [{"timestamp_sec": round(float(ev.timecode_sec), 2), "image_base64": ""}]
    times = ev.evidence_times[:MAX_EVIDENCE_FRAMES] or [ev.timecode_sec]
    bud = _budget.current()
    if bud is not None and bud.left_hard() < _budget.MIN_EVIDENCE_LEFT_SEC:
        bud.note("кадры-доказательства: по одному на нарушение (мало времени)")
        times = times[:1]
    caps = list(ev.evidence_captions) + [""] * len(times)
    rule = RULES[ev.type_code]
    frames: list[dict] = []
    # рамки вокруг объекта/зоны: ev.extras["evidence_boxes"] — список (x1, y1, x2, y2) или None на каждый кадр
    boxes = list(ev.extras.get("evidence_boxes") or []) + [None] * len(times)
    for ts, cap_text, xyxy in zip(times, caps, boxes):
        ts = max(0.0, min(float(ts), max(ctx.info.duration_sec - 0.05, 0.0)))
        frame = read_frame_at(ctx.path, ts, ctx.info)
        if frame is None:
            continue
        if xyxy is not None:
            try:
                frame = draw_box(frame, tuple(int(v) for v in xyxy), label=ev.type_code)
            except Exception:  # noqa: BLE001
                pass
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


def analyze_video(
    path: str,
    progress: Optional[ProgressFn] = None,
    deadline_ts: Optional[float] = None,
    hard_deadline_ts: Optional[float] = None,
    partial: Optional[Callable[[dict], None]] = None,
    light: bool = False,
) -> dict:
    """Анализ видео.

    deadline_ts / hard_deadline_ts — абсолютные time.time() мягкого и жёсткого дедлайнов (см. budget.py): без них
    пайплайн работает как раньше. partial(result) — вызывается, когда готов полноценный результат (правила + кадры),
    до необязательного VLM: воркер вернёт его, если жёсткий дедлайн наступит раньше. light=True — без слоя
    восприятия (только правила по яркости): запасной режим после сбоя процесса.
    """
    with _budget.scope(_budget.make_budget(deadline_ts, hard_deadline_ts)):
        return _analyze_video(path, progress, partial, light)


def _analyze_video(path: str, progress: Optional[ProgressFn], partial: Optional[Callable[[dict], None]], light: bool) -> dict:
    progress = progress or _noop
    t0 = time.time()
    bud = _budget.current()

    progress("decoding", 5)
    bt, bv, info = brightness_series(path, target_fps=10.0)
    ctx = VideoContext(path=path, info=info, bright_t=bt, bright_v=bv, progress=progress)
    log.info("decoded %s: %.1fs, %.2f fps, %dx%d", path, info.duration_sec, info.fps, info.width, info.height)

    # --- слой восприятия и состояний сцены (детекции людей/объектов, якоря, роли, зоны)
    perception_ok = False
    try:
        if light:
            raise RuntimeError("light mode: perception skipped")
        perc, states = _perceive(path, info, progress)
        from .scene import build_scene, scene_summary

        scene = build_scene(perc, states, info)
        ctx.scene = scene
        ctx.tracks = perc.tracks
        ctx.states = states
        ctx.zones = scene.anchors
        ctx.cache["scene_summary"] = scene_summary(scene)
        perception_ok = True
    except Exception as exc:  # noqa: BLE001 — без восприятия работают только правила по яркости (T9/T3-тёмный)
        if light:
            log.info("light mode: brightness-only rules")
        else:
            log.exception("perception layer failed, falling back to brightness-only rules: %s", exc)
        ctx.cache["perception_error"] = f"{type(exc).__name__}: {exc}"

    progress("rules", 60)
    # Фаза 1 — контекстные правила (T9): результат кладём в ctx.context_events ДО запуска остальных правил,
    # чтобы T1–T3 могли читать контекст (уход в темноте). Фаза 2 — правила-нарушения.
    context_dets = [d for d in DETECTORS if detector_code(d) in CONTEXT_ONLY_TYPES]
    violation_dets = [d for d in DETECTORS if detector_code(d) not in CONTEXT_ONLY_TYPES]
    context_events = _run_detectors(ctx, context_dets)
    ctx.context_events = context_events
    events = _run_detectors(ctx, violation_dets)
    # страховка: контекстные типы не становятся нарушениями, даже если их вернуло другое правило
    events = [e for e in events if e.type_code not in CONTEXT_ONLY_TYPES]

    progress("evidence", 80)
    violations = [event_to_violation(ctx, ev) for ev in events]

    def pack() -> dict:
        return _pack_result(info, violations, ctx, context_events, perception_ok, t0, bud, light)

    if partial is not None:
        try:
            partial(pack())
        except Exception as exc:  # noqa: BLE001
            log.warning("partial result callback failed: %s", exc)

    progress("describing", 95)
    # VLM-слой: уточнение описаний по кадрам события (ПЛАН 5.1, п. 4); без модели остаются шаблоны из правил
    if violations and bud is not None and bud.left_hard() < _budget.MIN_VLM_LEFT_SEC:
        bud.note("VLM-описания пропущены (мало времени)")
    elif violations:
        try:
            from .vlm import describe_violations

            describe_violations(path, violations, info)
        except Exception as exc:  # noqa: BLE001
            log.warning("vlm layer skipped: %s", exc)

    result = pack()
    progress("done", 100)
    return result


def _pack_result(info, violations: list, ctx: VideoContext, context_events: list, perception_ok: bool, t0: float, bud, light: bool) -> dict:
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
            "perception": perception_ok,
            "scene": ctx.cache.get("scene_summary"),
            "perception_error": ctx.cache.get("perception_error") if not light else None,
            "degraded": bool(bud is not None and bud.degraded) or light,
            "degrade_reasons": (list(bud.reasons) if bud is not None else []) + (["light mode: только правила по яркости"] if light else []),
            # контекст сцены (не нарушения): например, T9 — освещение выключено с t по end
            "context_events": [
                {
                    "type_code": e.type_code,
                    "name": RULES[e.type_code]["name"],
                    "timecode_sec": round(float(e.timecode_sec), 2),
                    "end_sec": None if e.end_sec is None else round(float(e.end_sec), 2),
                    "description": e.description,
                }
                for e in context_events
            ],
        },
    }
    if bud is not None and bud.stats:
        result["meta"]["degrade_stats"] = bud.stats
    return result


__all__ = ["analyze_video", "warmup", "PIPELINE_VERSION", "RULES"]
