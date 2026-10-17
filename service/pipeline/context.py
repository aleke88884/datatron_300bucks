"""Общий контекст одного видео, передаваемый всем детекторам.

Дорогие вычисления (декодирование, яркость, в будущем — детекции и треки) делаются один раз
в analyze_video и кладутся сюда; правила T1–T13 только читают состояния и порождают события.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from .video import VideoInfo

ProgressFn = Callable[[str, Optional[int]], None]


@dataclass
class Event:
    """Событие-нарушение, найденное правилом."""

    type_code: str                      # T1..T13
    timecode_sec: float                 # начало нарушающего действия / момент, когда условие стало истинным
    description: str                    # только наблюдаемое (ПЛАН 4.4): что видно и почему это нарушение
    confidence: float = 0.5
    end_sec: Optional[float] = None
    evidence_times: list[float] = field(default_factory=list)   # 1–3 момента для кадров-доказательств
    evidence_captions: list[str] = field(default_factory=list)  # подпись к каждому кадру (параллельно evidence_times)
    extras: dict[str, Any] = field(default_factory=dict)        # отладочные данные (не попадают в ответ API)


@dataclass
class VideoContext:
    path: str
    info: VideoInfo
    # ряд средней яркости кадра: times_sec, values 0..255 (~10 Гц)
    bright_t: np.ndarray
    bright_v: np.ndarray
    progress: ProgressFn = lambda stage, percent=None: None
    # --- слоты для слоёв восприятия/состояний (заполняет CV-команда) ---
    detections: Any = None        # per-frame bbox'ы людей/объектов
    tracks: Any = None            # треки людей с ролями (кассир/клиент/посторонний)
    states: Any = None            # per-second булевы состояния: cashier_in_room, safe_open, cash_on_desk, ...
    zones: Any = None             # зоны, построенные от детекций (стол, сейф, дверь, зона клиента)
    cache: dict[str, Any] = field(default_factory=dict)
