"""Подписи и рамки на кадрах-доказательствах. Кириллица рисуется через Pillow, если найден TTF-шрифт;
иначе — латинский fallback через cv2.putText (чтобы кадр никогда не оставался без подписи)."""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional, Sequence

import cv2
import numpy as np

FONT_CANDIDATES = [
    os.environ.get("CAPTION_FONT", ""),
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/arial.ttf",
]


@lru_cache(maxsize=1)
def _font_path() -> Optional[str]:
    for p in FONT_CANDIDATES:
        if p and os.path.exists(p):
            return p
    return None


def _wrap(draw, text: str, font, max_w: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    cur = ""
    for w in words:
        cand = (cur + " " + w).strip()
        if draw.textlength(cand, font=font) <= max_w or not cur:
            cur = cand
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines or [""]


def _draw_pil(frame: np.ndarray, lines: Sequence[str], color_bgr: tuple[int, int, int]) -> Optional[np.ndarray]:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return None
    fp = _font_path()
    if fp is None:
        return None
    h, w = frame.shape[:2]
    size = max(13, int(w * 0.036))
    try:
        font = ImageFont.truetype(fp, size)
    except OSError:
        return None
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img, "RGBA")
    pad = int(size * 0.5)
    bar = int(size * 0.5)
    x0 = pad + bar
    line_h = int(size * 1.3)
    # перенос строк по ширине кадра; первая строка — заголовок
    wrapped: list[tuple[str, bool]] = []
    for i, line in enumerate(lines):
        for part in _wrap(draw, line, font, w - x0 - pad):
            wrapped.append((part, i == 0))
    box_h = pad * 2 + line_h * len(wrapped)
    r, g, b = color_bgr[2], color_bgr[1], color_bgr[0]
    draw.rectangle([0, 0, w, box_h], fill=(0, 0, 0, 170))
    draw.rectangle([0, 0, bar, box_h], fill=(r, g, b, 255))
    y = pad
    for text, is_title in wrapped:
        draw.text((x0, y), text, font=font, fill=(r, g, b, 255) if is_title else (255, 255, 255, 255))
        y += line_h
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def _draw_cv(frame: np.ndarray, lines: Sequence[str], color_bgr: tuple[int, int, int]) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    scale = max(0.45, w / 900)
    line_h = int(28 * scale)
    box_h = 10 + line_h * len(lines)
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, out, 0.35, 0, out)
    cv2.rectangle(out, (0, 0), (6, box_h), color_bgr, -1)
    y = line_h
    for i, line in enumerate(lines):
        ascii_line = line.encode("ascii", "replace").decode("ascii")
        cv2.putText(out, ascii_line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, scale * 0.6, color_bgr if i == 0 else (255, 255, 255), 1, cv2.LINE_AA)
        y += line_h
    return out


def caption(frame: np.ndarray, lines: Sequence[str], color_bgr: tuple[int, int, int] = (60, 60, 230)) -> np.ndarray:
    """Баннер с подписью сверху кадра. lines[0] — заголовок (код типа), остальные — детали."""
    res = _draw_pil(frame, lines, color_bgr)
    if res is None:
        res = _draw_cv(frame, lines, color_bgr)
    return res


def box(frame: np.ndarray, xyxy: tuple[int, int, int, int], label: str = "", color_bgr: tuple[int, int, int] = (60, 60, 230)) -> np.ndarray:
    """Рамка вокруг зоны/объекта (для правил с детекциями)."""
    out = frame.copy()
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(out, (x1, y1), (x2, y2), color_bgr, 2)
    if label:
        cv2.putText(out, label.encode("ascii", "replace").decode("ascii"), (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 1, cv2.LINE_AA)
    return out
