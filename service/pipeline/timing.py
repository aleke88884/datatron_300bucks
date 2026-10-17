"""Уточнение таймкодов событий: «начало нарушающего действия» вместо момента, когда условие правила стало истинным.

Правила (rules_scene.py) решают, есть ли нарушение, по таблице состояний на сетке 2 к/с. Момент срабатывания
условия часто запаздывает относительно действия, которое отмечает разметка:

* уход кассира (T1): условие «кассир вне помещения» выполняется, когда трек пропал, а действие начинается, когда
  кассир заканчивает работу у стола/сейфа и направляется к выходу. Таймкод — конец последней остановки кассира
  в рабочей зоне перед уходом (refine_leave_onset);
* посторонний (T4): условие «три человека внутри» выполняется, когда новый трек впервые попал в зону «внутри»,
  иногда ещё за стеклом/в дверном проёме. Таймкод — момент, когда вошедший дошёл до места, где он остаётся
  (refine_entry_time);
* предмет на столе (T8): детектор на 3 к/с по полному кадру видит стакан позже, чем он появился. Перепроверяем
  несколько секунд до срабатывания по кропу стола с частотой 5 к/с (refine_object_onset).

Все пороги — в долях роста человека и в секундах (не в пикселях этой камеры). Уточнение меняет только
timecode_sec/evidence_times, условие нарушения (вердикт) не трогает.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("pipeline.timing")

# --- уход кассира ---------------------------------------------------------
STAT_RADIUS = 0.10        # остановка: стопа смещается не более чем на 0.10 роста от среднего положения…
STAT_RADIUS_FIRST = 0.15  # …и не более чем на 0.15 роста от первой точки (медленный дрейф — это ходьба)
STAT_HEIGHT_TOL = 0.15    # рост рамки меняется не более чем на 15 % (вставание/наклон — это действие)
STAT_GAP_SEC = 1.0        # разрыв трека больше 1 с обрывает остановку
STAT_MIN_SEC = 1.5        # остановка короче 1.5 с — пауза на ходу (поворот, открывание двери), не работа
WORK_MIN_SEC = 2.5        # «рабочее место» — где кассир стоял/сидел не меньше 2.5 с до начала ухода…
WORK_RADIUS = 0.30        # …остановка перед уходом считается работой, если она не дальше 0.3 роста от такого места
LEAVE_LOOKBACK_SEC = 12.0  # уход ищем не раньше чем за 12 с до исчезновения кассира

# --- посторонний ----------------------------------------------------------
ENTRY_WINDOW_SEC = 6.0    # путь вошедшего анализируем 6 с после появления внутри
ENTRY_REACH = 0.8         # момент, когда пройдено 80 % наибольшего удаления от точки входа

# --- предмет на столе -----------------------------------------------------
OBJ_LOOKBACK_SEC = 4.0
OBJ_FPS = 5.0
OBJ_MIN_SCORE = 0.25
OBJ_CONSEC = 3            # предмет виден в 3 отсчётах подряд (0.6 с) — не случайная детекция


@dataclass
class Stop:
    t0: float                 # первая точка остановки
    t1: float                 # последняя точка остановки
    t_next: Optional[float]   # первая точка после остановки (начало движения)
    foot: tuple[float, float]
    h: float

    @property
    def dur(self) -> float:
        return self.t1 - self.t0


def cashier_obs(sc, tracks) -> list:
    """Наблюдения всех треков кассира (по времени)."""
    ids = set(getattr(sc, "cashier_track_ids", None) or [])
    if not ids or not tracks:
        return []
    obs = [o for tr in tracks if tr.id in ids for o in tr.obs]
    obs.sort(key=lambda o: o.t)
    # в один момент — одно наблюдение (самая крупная рамка)
    out = []
    for o in obs:
        if out and abs(out[-1].t - o.t) < 1e-3:
            if o.h > out[-1].h:
                out[-1] = o
            continue
        out.append(o)
    return out


def stops(obs: list) -> list[Stop]:
    """Разбиение траектории на остановки (жадно, слева направо)."""
    res: list[Stop] = []
    n = len(obs)
    i = 0
    while i < n:
        first = obs[i]
        pts = [first]
        j = i + 1
        while j < n:
            o = obs[j]
            if o.t - pts[-1].t > STAT_GAP_SEC:
                break
            href = float(np.median([p.h for p in pts])) or 1.0
            mx = float(np.mean([p.foot[0] for p in pts]))
            my = float(np.mean([p.foot[1] for p in pts]))
            d_mean = np.hypot(o.foot[0] - mx, o.foot[1] - my) / href
            d_first = np.hypot(o.foot[0] - first.foot[0], o.foot[1] - first.foot[1]) / href
            if d_mean > STAT_RADIUS or d_first > STAT_RADIUS_FIRST or abs(o.h / href - 1.0) > STAT_HEIGHT_TOL:
                break
            pts.append(o)
            j += 1
        res.append(Stop(
            t0=pts[0].t, t1=pts[-1].t, t_next=obs[j].t if j < n else None,
            foot=(float(np.mean([p.foot[0] for p in pts])), float(np.mean([p.foot[1] for p in pts]))),
            h=float(np.median([p.h for p in pts])),
        ))
        i = j
    return res


def refine_leave_onset(sc, tracks, t_gone: float) -> Optional[float]:
    """Начало ухода кассира: момент, когда кассир трогается с последней остановки в рабочей зоне перед
    исчезновением (t_gone — начало участка «ушёл»; ищем не раньше LEAVE_LOOKBACK_SEC до него).
    None — уточнить нельзя (оставить исходный таймкод).

    Остановка «в рабочей зоне» — длиннее STAT_MIN_SEC и рядом (≤ WORK_RADIUS роста) с местом, где кассир
    провёл не меньше WORK_MIN_SEC раньше (сидел за столом, стоял у сейфа). Короткие паузы на пути к выходу
    (открывает дверь, оборачивается) и остановки у двери/за стеклом сюда не попадают.
    """
    obs = cashier_obs(sc, tracks)
    if len(obs) < 4:
        return None
    all_stops = stops(obs)
    lo = max(0.0, t_gone - LEAVE_LOOKBACK_SEC)
    work = [s for s in all_stops if s.dur >= WORK_MIN_SEC and s.t1 < t_gone]
    best: Optional[Stop] = None
    for s in all_stops:
        if s.dur < STAT_MIN_SEC or s.t_next is None or s.t1 < lo or s.t_next > t_gone:
            continue
        others = [w for w in work if w is not s and w.t0 < s.t0]
        anchor = others + ([s] if s.dur >= WORK_MIN_SEC else [])
        if not anchor:
            continue
        near = min(np.hypot(s.foot[0] - w.foot[0], s.foot[1] - w.foot[1]) / max(s.h, w.h, 1.0) for w in anchor)
        if near <= WORK_RADIUS:
            best = s
    if best is None:
        return None
    return float(best.t_next)


def cashier_box_at(sc, tracks, t: float, tol: float = 0.4) -> Optional[tuple[int, int, int, int]]:
    obs = cashier_obs(sc, tracks)
    if not obs:
        return None
    o = min(obs, key=lambda o: abs(o.t - t))
    if abs(o.t - t) > tol:
        return None
    return tuple(int(v) for v in o.xyxy)


# --- T4: вход постороннего -------------------------------------------------

def newcomer_track(sc, tracks, t_start: float):
    """Трек вошедшего: среди уверенных посетителей, которые внутри в момент t_start, — появившийся последним."""
    vis = set(getattr(sc, "visitor_track_ids", None) or [])
    best = None
    for tr in tracks or []:
        if tr.id not in vis:
            continue
        inside = [o for o in tr.obs if o.inside]
        if not inside or not any(abs(o.t - t_start) <= 0.6 for o in inside):
            continue
        if best is None or inside[0].t > best[1]:
            best = (tr, inside[0].t)
    return best[0] if best else None


def refine_entry_time(sc, tracks, t_start: float, t_end: float) -> Optional[tuple[float, int]]:
    """Момент, когда вошедший прошёл ENTRY_REACH своего наибольшего удаления от точки, где его впервые
    засчитали «внутри» (стоит в проёме → дошёл до стола/кассира). Возвращает (время, id трека)."""
    tr = newcomer_track(sc, tracks, t_start)
    if tr is None:
        return None
    win = [o for o in tr.obs if t_start - 0.6 <= o.t <= min(t_end, t_start + ENTRY_WINDOW_SEC) + 0.01]
    if len(win) < 3:
        return None
    p0 = win[0].foot
    href = float(np.median([o.h for o in win])) or 1.0
    d = np.asarray([np.hypot(o.foot[0] - p0[0], o.foot[1] - p0[1]) / href for o in win])
    if d.max() < 0.25:          # не перемещался: стоит там, где вошёл
        return None
    k = int(np.flatnonzero(d >= ENTRY_REACH * d.max())[0])
    return max(t_start, float(win[k].t)), tr.id


def track_box_at(tracks, tid: int, t: float, tol: float = 0.4) -> Optional[tuple[int, int, int, int]]:
    for tr in tracks or []:
        if tr.id == tid:
            o = tr.at(t, tol)
            return tuple(int(v) for v in o.xyxy) if o else None
    return None


# --- T8: появление предмета на столе ---------------------------------------

def _cache_file(path: str) -> Optional[Path]:
    root = os.environ.get("FEATURES_CACHE")
    if not root:
        return None
    stem = Path(path).stem.replace("X", "Х")
    return Path(root) / "timing" / f"{stem}.json"


def _dense_detections(path: str, info, t0: float, t1: float, zone: tuple[int, int, int, int], classes: set[str]) -> list:
    """[(t, [(cls, score, xyxy в координатах кадра), ...]), ...] по кропу zone с частотой OBJ_FPS."""
    from .detect import get_detector
    from .video import iter_frames

    det = get_detector()
    step = max(1, int(round(info.fps / OBJ_FPS)))
    x1, y1, x2, y2 = zone
    out = []
    for idx, t, frame in iter_frames(path, every_n=step):
        if t < t0 - 1e-6:
            continue
        if t > t1 + 1e-6:
            break
        crop = frame[y1:y2, x1:x2]
        ds = det(crop, conf=0.15, classes=classes)
        out.append((round(float(t), 3), [(d.cls, round(float(d.score), 3),
                                          [round(d.xyxy[0] + x1, 1), round(d.xyxy[1] + y1, 1), round(d.xyxy[2] + x1, 1), round(d.xyxy[3] + y1, 1)])
                                         for d in ds]))
    return out


def refine_object_onset(path: str, info, t_hit: float, zone, classes: set[str]) -> Optional[tuple[float, tuple]]:
    """Первое появление предмета (classes) в зоне стола: перепроверка OBJ_LOOKBACK_SEC до срабатывания
    по кропу стола на OBJ_FPS. Предмет засчитывается, если виден OBJ_CONSEC отсчётов подряд со score ≥ OBJ_MIN_SCORE.
    Возвращает (время, bbox) или None. При FEATURES_CACHE детекции кешируются в results/timing/<видео>.json."""
    H, W = info.height, info.width
    z = (max(0, int(zone[0])), max(0, int(zone[1])), min(W, int(zone[2])), min(H, int(zone[3])))
    if z[2] - z[0] < 16 or z[3] - z[1] < 16:
        return None
    t0, t1 = max(0.0, t_hit - OBJ_LOOKBACK_SEC), t_hit + 0.5
    key = {"t0": round(t0, 2), "t1": round(t1, 2), "zone": list(z), "classes": sorted(classes), "fps": OBJ_FPS}
    cf = _cache_file(path)
    data = None
    store: dict = {}
    if cf is not None and cf.exists():
        try:
            store = json.loads(cf.read_text(encoding="utf-8"))
            ent = store.get("t8_dense")
            if ent and ent.get("key") == key:
                data = ent["dets"]
        except Exception:  # noqa: BLE001
            store = {}
    if data is None:
        try:
            data = _dense_detections(path, info, t0, t1, z, classes)
        except Exception as exc:  # noqa: BLE001 — нет детектора/видео: оставляем исходный таймкод
            log.warning("dense object re-detection failed: %s", exc)
            return None
        if cf is not None:
            try:
                cf.parent.mkdir(parents=True, exist_ok=True)
                store["t8_dense"] = {"key": key, "dets": data}
                cf.write_text(json.dumps(store, ensure_ascii=False), encoding="utf-8")
            except OSError:
                pass
    run: list = []
    for t, ds in data:
        good = [d for d in ds if d[0] in classes and d[1] >= OBJ_MIN_SCORE]
        if good:
            run.append((t, max(good, key=lambda d: d[1])))
            if len(run) >= OBJ_CONSEC:
                t_on, d = run[0]
                return float(t_on), tuple(int(v) for v in d[2])
        else:
            run = []
    return None


# --- T5/T4: где стоит посетитель --------------------------------------------

DESK_FOOT_PAD = 0.15      # стопа «у стола»: внутри рамки стола, расширенной на 0.15 роста человека


def track_first_inside(tracks, tid: int) -> Optional[float]:
    for tr in tracks or []:
        if tr.id == tid:
            ins = [o.t for o in tr.obs if o.inside]
            return float(ins[0]) if ins else None
    return None


def visitors_at_desk_frac(sc, tracks, t0: float, t1: float) -> Optional[tuple[float, int]]:
    """Для посетителя, дольше всех видимого внутри в [t0; t1]: доля его наблюдений, когда стопа у стола
    (в рамке стола, расширенной на DESK_FOOT_PAD его роста). (доля, id трека) или None, если не посчитать."""
    desk = (getattr(sc, "anchors", None) or {}).get("desk")
    vis = set(getattr(sc, "visitor_track_ids", None) or [])
    if not desk or not vis or not tracks:
        return None
    x1, y1, x2, y2 = desk["bbox"]
    best = None
    for tr in tracks:
        if tr.id not in vis:
            continue
        obs = [o for o in tr.obs if o.inside and t0 - 0.01 <= o.t <= t1 + 0.01]
        if len(obs) < 2:
            continue
        near = 0
        for o in obs:
            pad = DESK_FOOT_PAD * o.h
            fx, fy = o.foot
            if x1 - pad <= fx <= x2 + pad and y1 - pad <= fy <= y2 + pad:
                near += 1
        frac = near / len(obs)
        if best is None or len(obs) > best[2]:
            best = (frac, tr.id, len(obs))
    return (best[0], best[1]) if best else None
