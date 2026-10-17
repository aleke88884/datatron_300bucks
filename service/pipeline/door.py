"""Состояние двери кассового помещения: самокалибрующаяся область двери и ряд «насколько дверь отличается от закрытой».

Никаких пиксельных зон под конкретную камеру. Дверь находим по поведению:
  1. Входы/выходы — моменты, когда уверенный трек человека впервые/в последний раз виден внутри кассы (perception).
  2. Вокруг каждого входа/выхода копим движение между соседними кадрами ВНЕ рамок людей: двигаться там может только
     то, что двигает человек, — створка двери. Крупнейшая связная область движения рядом с человеком — кандидат.
     Кандидаты объединяются по IoU; область двери — устойчивый кластер (чаще всего повторяющийся, затем самый крупный).
  3. Эталон «до двери» — медиана области за ~3 с до первого движения створки в ролике (рамки людей исключены).
  4. По кадрам (4 к/с): доля пикселей области (без людей), отличающихся от эталона больше чем на CHANGE_THR уровней
     яркости после выравнивания среднего (компенсация автоэкспозиции). 0 — дверь как в эталоне (закрыта),
     ~пик первого открывания — дверь открыта.
  5. Пересечения дверного проёма: для каждого входа/выхода — момент наибольшего перекрытия области двери рамкой
     этого человека (человек в проёме).

Выход — только числа (без изображений), кешируется в results/door/<видео>.json (scripts/collect_door.py) и читается
из кеша только при заданном FEATURES_CACHE; в проде всё считается по видео. Решение «дверь оставлена открытой»
принимает pipeline/rules_door.py.
"""
from __future__ import annotations

import json
import logging
import os
import warnings
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .perception import Perception
from .video import VideoInfo, iter_frames

log = logging.getLogger("pipeline.door")

DOOR_VERSION = 1
DOOR_FPS = 4.0             # частота кадров для двери (движение створки длится 1–3 с)
WORK_SIDE = 424            # рабочее разрешение: длинная сторона кадра, пикс. (464x848 → 0.5)
MOTION_THR = 15            # |кадр − предыдущий| > 15 уровней яркости — движение
CHANGE_THR = 25            # |кадр − эталон| > 25 уровней (после выравнивания среднего) — пиксель изменился
PERSON_PAD = 0.2           # рамка человека расширяется на 20 % (движение, тени, неточность рамки)
MASK_TOL_SEC = 0.4         # кадр двери берёт рамки людей из кадров восприятия в пределах ±0.4 с
MIN_TRACK_INSIDE = 5       # уверенный трек: ≥ 5 наблюдений внутри…
MIN_TRACK_SCORE = 0.6      # …с медианной уверенностью ≥ 0.6
ENTRY_WINDOW = (-2.0, 5.0) # окно движения створки вокруг входа (открыть до шага, закрыть после)
EXIT_WINDOW = (-5.0, 2.0)  # и вокруг выхода
COMP_MAX_DIST = 1.5        # центр области движения не дальше 1.5 высоты человека от его рамки
COMP_MIN_AREA = 0.5        # и площадью не меньше 50 % рамки человека (створка — крупный объект, не рука/тень/стул)
COMP_MIN_H = 0.8           # и высотой не меньше 80 % роста человека (дверь не ниже того, кто в неё проходит)
REF_SEC = 3.0              # эталон — медиана за 3 с до первого движения створки
MOTION_START = 0.05        # «створка пошла»: ≥ 5 % пикселей области двигаются
MIN_VISIBLE = 0.3          # доля области, не закрытая людьми, чтобы кадр считался


def _work_scale(info: VideoInfo) -> float:
    return min(1.0, WORK_SIDE / float(max(info.width, info.height, 1)))


def _decode(path: str, info: VideoInfo, scale: float) -> tuple[np.ndarray, np.ndarray]:
    step = max(1, int(round(info.fps / DOOR_FPS)))
    w, h = max(1, int(round(info.width * scale))), max(1, int(round(info.height * scale)))
    ts, frames = [], []
    for _, t, frame in iter_frames(path, every_n=step):
        g = cv2.cvtColor(cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        ts.append(t)
        frames.append(g)
    if not frames:
        return np.zeros(0), np.zeros((0, h, w), np.uint8)
    return np.asarray(ts, float), np.stack(frames)


def _person_masks(perc: Perception, times: np.ndarray, shape: tuple[int, int], scale: float) -> np.ndarray:
    H, W = shape
    ptimes = np.asarray(perc.times, float) if perc.times else np.zeros(0)
    out = np.zeros((len(times), H, W), bool)
    for i, t in enumerate(times):
        if len(ptimes) == 0:
            continue
        for j in np.flatnonzero(np.abs(ptimes - t) <= MASK_TOL_SEC):
            for o in perc.persons[j]:
                x1, y1, x2, y2 = (v * scale for v in o.xyxy)
                w, h = x2 - x1, y2 - y1
                a = int(max(0, x1 - PERSON_PAD * w))
                b = int(max(0, y1 - PERSON_PAD * h))
                c = int(min(W, x2 + PERSON_PAD * w))
                d = int(min(H, y2 + 0.5 * PERSON_PAD * h))
                out[i, b:d, a:c] = True
    return out


def _candidates(perc: Perception, duration: float) -> list[dict]:
    """Входы и выходы уверенных треков. Человек, уже неподвижно сидящий в кадре с первых секунд, — не вход."""
    out = []
    for tr in perc.tracks:
        ins = [o for o in tr.obs if o.inside]
        if len(ins) < MIN_TRACK_INSIDE or float(np.median([o.score for o in ins])) < MIN_TRACK_SCORE:
            continue
        f, l = ins[0], ins[-1]
        static_start = False
        if f.t < 0.5:
            cx = np.asarray([(o.xyxy[0] + o.xyxy[2]) / 2 for o in ins if o.t < 1.6])
            static_start = len(cx) > 0 and float(np.ptp(cx)) < 0.15 * f.h
        if not static_start:
            out.append({"kind": "in", "t": float(f.t), "track": int(tr.id), "xyxy": tuple(float(v) for v in f.xyxy)})
        if l.t < duration - 1.0:
            out.append({"kind": "out", "t": float(l.t), "track": int(tr.id), "xyxy": tuple(float(v) for v in l.xyxy)})
    return sorted(out, key=lambda c: c["t"])


def _motion_component(t: np.ndarray, g: np.ndarray, masks: np.ndarray, cand: dict, scale: float) -> Optional[dict]:
    lo, hi = ENTRY_WINDOW if cand["kind"] == "in" else EXIT_WINDOW
    sel = np.flatnonzero((t >= cand["t"] + lo) & (t <= cand["t"] + hi))
    H, W = g.shape[1:]
    acc = np.zeros((H, W), np.uint16)
    for i in sel:
        if i == 0:
            continue
        d = cv2.absdiff(g[i], g[i - 1]) > MOTION_THR
        acc += (d & ~masks[i] & ~masks[i - 1]).astype(np.uint16)
    mm = (acc >= 2).astype(np.uint8)
    k_close = max(3, int(round(7 * WORK_SIDE / 424)) | 1)
    mm = cv2.morphologyEx(mm, cv2.MORPH_CLOSE, np.ones((k_close, k_close), np.uint8))
    mm = cv2.morphologyEx(mm, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(mm)
    x1, y1, x2, y2 = (v * scale for v in cand["xyxy"])
    bh = max(1.0, y2 - y1)
    bcx, bcy = (x1 + x2) / 2, (y1 + y2) / 2
    best = None
    for j in range(1, n):
        if np.hypot(cen[j][0] - bcx, cen[j][1] - bcy) / bh > COMP_MAX_DIST:
            continue
        # створка двери не ниже проходящего через неё человека и сопоставима с ним по площади
        # (на датасете: высота 0.9–2.8 рамки, площадь 0.7–7.6 рамки); сдвинутый стул/сумка — заметно меньше
        if st[j][3] < COMP_MIN_H * bh or st[j][4] < COMP_MIN_AREA * (x2 - x1) * bh:
            continue
        if best is None or st[j][4] > st[best][4]:
            best = j
    if best is None:
        return None
    x, y, w, h, a = (int(v) for v in st[best])
    return {"roi": (x, y, x + w, y + h), "mask": lab[y:y + h, x:x + w] == best, "area": a, **cand}


def _iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _not_found(reason: str, scale: float) -> dict:
    return {"version": DOOR_VERSION, "fps": DOOR_FPS, "scale": scale, "found": False, "reason": reason}


def compute_door(path: str, info: VideoInfo, perc: Perception, progress=None) -> dict:
    """Числовые признаки двери по видео (см. докстринг модуля)."""
    scale = _work_scale(info)
    cands = _candidates(perc, info.duration_sec)
    if not cands:
        return _not_found("нет входов/выходов уверенных треков", scale)
    t, g = _decode(path, info, scale)
    if len(t) < 4:
        return _not_found("мало кадров", scale)
    if progress:
        progress("door", 50)
    masks = _person_masks(perc, t, g.shape[1:], scale)
    comps = [c for c in (_motion_component(t, g, masks, c, scale) for c in cands) if c]
    if not comps:
        return _not_found("нет движения створки рядом с входами/выходами", scale)
    clusters: list[list[dict]] = []
    for c in comps:
        for cl in clusters:
            if _iou(cl[0]["roi"], c["roi"]) >= 0.3:
                cl.append(c)
                break
        else:
            clusters.append([c])
    cl = max(clusters, key=lambda cl: (len(cl), max(c["area"] for c in cl)))
    best = max(cl, key=lambda c: c["area"])
    x1, y1, x2, y2 = best["roi"]
    reg = best["mask"]
    n = len(t)
    gr = g[:, y1:y2, x1:x2].astype(np.int16)
    mr = masks[:, y1:y2, x1:x2]
    # движение внутри области двери: первый момент, когда створка пошла
    mot = np.zeros(n)
    for i in range(1, n):
        v = reg & ~mr[i] & ~mr[i - 1]
        if v.sum() < 0.2 * reg.sum():
            continue
        mot[i] = float((np.abs(gr[i] - gr[i - 1])[v] > MOTION_THR).mean())
    moving = np.flatnonzero(mot >= MOTION_START)
    if not len(moving):
        return _not_found("створка не двигается", scale)
    tm0 = float(t[moving[0]])
    rsel = np.flatnonzero((t >= tm0 - REF_SEC) & (t <= tm0 - 0.25))
    if len(rsel) < 2:
        return _not_found("дверь двигается с начала записи — нет эталона", scale)
    stack = np.where(reg[None] & ~mr[rsel], gr[rsel].astype(np.float32), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # пиксели, закрытые людьми во всех кадрах эталона → NaN
        ref = np.nanmedian(stack, axis=0)
    ref_ok = ~np.isnan(ref)
    ref_bright = float(np.mean([g[i].mean() for i in rsel]))
    change: list[Optional[float]] = []
    bright_ratio: list[float] = []
    for i in range(n):
        bright_ratio.append(round(float(g[i].mean()) / max(ref_bright, 1e-3), 3))
        v = reg & ~mr[i] & ref_ok
        if v.sum() < MIN_VISIBLE * reg.sum():
            change.append(None)
            continue
        cur = gr[i].astype(np.float32)
        dd = (cur - np.median(cur[v])) - (ref - np.median(ref[v]))
        change.append(round(float((np.abs(dd[v]) > CHANGE_THR).mean()), 4))
    burst = [c for c, tt in zip(change, t) if c is not None and tm0 <= tt <= tm0 + 3.0]
    peak = float(np.percentile(burst, 90)) if burst else None
    # пересечения проёма: момент наибольшего перекрытия области двери рамкой человека этого трека
    roi_full = (x1 / scale, y1 / scale, x2 / scale, y2 / scale)
    area = max(1.0, (roi_full[2] - roi_full[0]) * (roi_full[3] - roi_full[1]))
    tracks = {tr.id: tr for tr in perc.tracks}
    crossings = []
    for c in cands:
        lo, hi = ENTRY_WINDOW if c["kind"] == "in" else EXIT_WINDOW
        best_cov, best_t = 0.0, None
        for o in tracks[c["track"]].obs:
            if not (c["t"] + lo <= o.t <= c["t"] + hi):
                continue
            ix = max(0.0, min(roi_full[2], o.xyxy[2]) - max(roi_full[0], o.xyxy[0]))
            iy = max(0.0, min(roi_full[3], o.xyxy[3]) - max(roi_full[1], o.xyxy[1]))
            cov = ix * iy / area
            if cov > best_cov:
                best_cov, best_t = cov, float(o.t)
        crossings.append({"kind": c["kind"], "t": round(c["t"], 2), "track": c["track"],
                          "t_cross": None if best_t is None else round(best_t, 2), "cover": round(best_cov, 3)})
    return {
        "version": DOOR_VERSION, "fps": DOOR_FPS, "scale": round(scale, 4), "found": True,
        "roi": [round(v, 1) for v in roi_full],
        "roi_frac": round(area / float(info.width * info.height), 4),
        "members": [{"kind": c["kind"], "t": round(c["t"], 2), "track": c["track"]} for c in cl],
        "t_first_motion": round(tm0, 2), "ref_window": [round(float(t[rsel[0]]), 2), round(float(t[rsel[-1]]), 2)],
        "peak": None if peak is None else round(peak, 4),
        "t": [round(float(v), 2) for v in t], "change": change, "bright_ratio": bright_ratio,
        "crossings": crossings,
    }


def door_features(path: str, info: VideoInfo, perc: Perception, progress=None) -> dict:
    """Признаки двери: из results/door/<видео>.json при заданном FEATURES_CACHE (и записи туда, если файла нет),
    иначе — по видео."""
    cache = os.environ.get("FEATURES_CACHE")
    cj = None
    if cache:
        stem = Path(path).stem.replace("X", "Х")
        cj = Path(cache) / "door" / f"{stem}.json"
        if cj.exists():
            try:
                d = json.loads(cj.read_text(encoding="utf-8"))
                if d.get("version") == DOOR_VERSION:
                    return d
            except Exception as exc:  # noqa: BLE001
                log.warning("door cache %s unreadable: %s", cj, exc)
    d = compute_door(path, info, perc, progress=progress)
    if cj is not None:
        cj.parent.mkdir(parents=True, exist_ok=True)
        cj.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    return d


def door_summary(d: Optional[dict]) -> dict:
    """Компактная сводка для meta.scene ответа."""
    if not d:
        return {"found": False}
    if not d.get("found"):
        return {"found": False, "reason": d.get("reason")}
    return {"found": True, "roi": d["roi"], "t_first_motion": d["t_first_motion"], "peak": d["peak"],
            "crossings": len(d.get("crossings", []))}
