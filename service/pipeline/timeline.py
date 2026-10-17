"""Дорожки состояний сцены для UI (объяснимость вердикта): интервалы времени, когда кассир в кассе,
есть посетители, сейф открыт, на столе предметы, экран активен, свет выключен.

Строится из Scene теми же порогами, что и правила (rules_scene.py, scene_summary), поэтому дорожки не могут
противоречить событиям: «сейф открыт» = safe_open_score ≥ 1.0, «предметы на столе» = desk_obj_area ≥ DESK_OBJ_MIN
или инкассаторская сумка, «экран активен» = screen == 'active'. Неизвестное состояние (сейф перекрыт, экран не
найден) выдаётся отдельной дорожкой *_unknown, а не как «закрыт/выключен».

При выключенном свете детекции людей/предметов и цветовой классификатор экрана ненадёжны (правила T1/T2 в темноте
тоже молчат), поэтому в эти интервалы дорожки состояний пусты, сейф/экран/стол помечены «неизвестно», а сам
интервал выдаётся как "unreliable" — UI штрихует его поверх всех дорожек.

Формат (только числа, ~сотни байт):
    {"step": 0.5, "duration": D,
     "anchors": {"desk": bool, "safe": bool, "laptop": bool},
     "lanes": {"cashier": [[a, b], ...], "at_desk": [...], "visitors": [[a, b, n], ...],
               "safe_open": [...], "safe_unknown": [...], "desk_items": [...], "bag": [...], "desk_unknown": [...],
               "screen_active": [...], "screen_locked": [...], "screen_unknown": [...], "lights_off": [...]},
     "unreliable": [[a, b], ...]}
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .scene import DESK_OBJ_MIN, GRID_FPS, Scene

SAFE_OPEN_SCORE = 1.0      # тот же порог, что T2_SCORE в rules_scene.py и safe_open_frames в scene_summary
FILL_GAP_SEC = 0.5         # одиночный «провал» сетки (0.5 с) — шум детектора, на дорожке не показываем


def _r(x: float) -> float:
    return round(float(x), 1)


def _intervals(sc: Scene, mask, duration: float) -> list[list[float]]:
    """Маска на сетке sc.t → [[t_start, t_end]], каждая ячейка сетки покрывает [t, t + 1/GRID_FPS)."""
    mask = np.asarray(mask, bool)
    if mask.size == 0 or not mask.any():
        return []
    dt = 1.0 / GRID_FPS
    out = []
    for a, b in sc.runs(mask, min_sec=dt, fill_gap_sec=FILL_GAP_SEC):
        t0 = max(0.0, float(sc.t[a]))
        t1 = min(duration, float(sc.t[b]) + dt) if duration > 0 else float(sc.t[b]) + dt
        if t1 > t0:
            out.append([_r(t0), _r(t1)])
    return out


def _count_intervals(sc: Scene, counts, duration: float) -> list[list[float]]:
    """Run-length по числу посетителей (> 0): [[t_start, t_end, n], ...]; одиночные провалы сглаживаются."""
    c = np.asarray(counts, int).copy()
    n = len(c)
    if n == 0 or not (c > 0).any():
        return []
    gap = int(round(FILL_GAP_SEC * GRID_FPS))
    # провал в 0 длиной ≤ gap между ненулевыми — заполняем соседним значением
    i = 0
    while i < n:
        if c[i] == 0:
            j = i
            while j < n and c[j] == 0:
                j += 1
            if 0 < i and j < n and (j - i) <= gap:
                c[i:j] = min(c[i - 1], c[j])
            i = j
        else:
            i += 1
    dt = 1.0 / GRID_FPS
    out: list[list[float]] = []
    start = None
    for k in range(n + 1):
        v = int(c[k]) if k < n else 0
        if start is not None and (k == n or v != int(c[start])):
            t0 = max(0.0, float(sc.t[start]))
            t1 = float(sc.t[k - 1]) + dt
            if duration > 0:
                t1 = min(duration, t1)
            if t1 > t0:
                out.append([_r(t0), _r(t1), int(c[start])])
            start = None
        if start is None and k < n and v > 0:
            start = k
    return out


def build_timeline(scene: Scene, duration: Optional[float] = None) -> dict:
    sc = scene
    dur = float(duration if duration is not None else (sc.info.duration_sec or 0.0))
    if dur <= 0 and len(sc.t):
        dur = float(sc.t[-1]) + 1.0 / GRID_FPS
    n = len(sc.t)

    safe_known = np.asarray([v is not None for v in sc.safe_open_score], bool) if n else np.zeros(0, bool)
    safe_score = np.asarray([v if v is not None else 0.0 for v in sc.safe_open_score], float) if n else np.zeros(0)
    screen = list(sc.screen) if n else []
    anchors = sc.anchors or {}

    lit = ~np.asarray(sc.lights_off, bool) if n else np.zeros(0, bool)
    desk_items = (np.asarray(sc.desk_obj_area) >= DESK_OBJ_MIN) | np.asarray(sc.bag_on_desk, bool)
    screen_known = np.asarray([s is not None for s in screen], bool)

    def lane(mask) -> list[list[float]]:
        return _intervals(sc, np.asarray(mask, bool) & lit, dur)

    def unknown(known, anchor: str) -> list[list[float]]:
        if not anchors.get(anchor):
            return []    # объект не найден в сцене вообще — UI пишет «не обнаружен» по anchors
        return _intervals(sc, ~np.asarray(known, bool) | ~lit, dur)

    lanes = {
        "cashier": lane(sc.cashier_inside),
        "at_desk": lane(sc.cashier_at_desk),
        "visitors": _count_intervals(sc, np.where(lit, sc.n_visitors, 0), dur),
        "safe_open": lane(safe_known & (safe_score >= SAFE_OPEN_SCORE)),
        "safe_unknown": unknown(safe_known, "safe"),
        "desk_items": lane(desk_items),
        "bag": lane(sc.bag_on_desk),
        "desk_unknown": unknown(np.ones(n, bool), "desk"),
        "screen_active": lane([s == "active" for s in screen]),
        "screen_locked": lane([s == "locked" for s in screen]),
        "screen_unknown": unknown(screen_known, "laptop"),
        "lights_off": _intervals(sc, sc.lights_off, dur),
    }
    return {
        "step": 1.0 / GRID_FPS,
        "duration": _r(dur),
        "anchors": {k: bool(anchors.get(k)) for k in ("desk", "safe", "laptop")},
        "lanes": lanes,
        "unreliable": lanes["lights_off"],
    }


__all__ = ["build_timeline"]
