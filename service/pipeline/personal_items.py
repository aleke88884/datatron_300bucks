"""T8 — личные вещи, еда и напитки на рабочем месте кассира (Правила № 231, п. 12).

Два наблюдаемых сценария, оба — по детекциям YOLOX из слоя восприятия (3 к/с) и ролям из scene.py:

1. «Напиток/еда у кассира» — стакан/бутылка/бокал (или еда) уверенно распознаны в зоне стола много раз
   (стабильный предмет, а не единичная ошибка детектора). Момент появления уточняется назад по времени:
   детектор запускается на увеличенном кропе зоны стола (мелкий стакан на полном кадре 464×848 YOLOX
   часто не видит, на кропе ×2 — видит), и таймкод = первый кадр, где напиток уже в руке/на столе.
   Пример: Х-46 (полный кадр — стакан с 13.0 с, кроп — с ~11 с; разметка 11 с).

2. «Кассир вносит личную сумку» — сумка/рюкзак/чемодан в руках КАССИРА (рамка пересекается с рамкой
   трека-кассира), предмет движется вместе с ним (не статичный предмет интерьера, который детектор путает
   с сумкой: стул, урна — их рамка неподвижна и видна без людей рядом) и не является зелёной
   инкассаторской сумкой (Х-14, Х-15, Х-17 — служебная сумка, не личная вещь). Таймкод — момент, когда
   кассир с внесённой сумкой занимает рабочее место (садится за стол): с этого момента личные вещи
   находятся на рабочем месте; если кассир не садится — первый кадр с сумкой. Пример: Х-40.

Пиксельные признаки (доля зелёного в рамке сумки, детекции на кропе стола) считаются по видео только для
кандидатов (в чистых роликах обычно ни одного чтения кадра) и, если задан FEATURES_CACHE, запоминаются в
results/personal_items/<видео>.json (только числа) — офлайн-оценка не декодирует видео повторно.
Все пороги относительные (доли размеров рамок/стола, секунды), без пиксельных зон под эту камеру.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .context import Event, VideoContext

log = logging.getLogger("pipeline.personal_items")

ITEMS_CACHE_VERSION = 1

DRINK_CLASSES = ("cup", "bottle", "wine glass")
FOOD_CLASSES = ("sandwich", "banana", "apple", "orange", "donut", "cake", "pizza", "hot dog")
BAG_CLASSES = ("handbag", "backpack", "suitcase")

# --- напиток / еда ---------------------------------------------------------
DRINK_MIN_DET = 4            # детекций напитка в зоне стола за ролик (3 к/с)
DRINK_STRONG_SCORE = 0.5     # «уверенная» детекция
DRINK_MIN_STRONG = 3         # уверенных детекций не меньше трёх (единичные 0.5+ бывают на интерьере, Х-19)
ZONE_UP = 0.8                # зона стола: bbox столешницы + 0.8 её высоты вверх (предмет в руке над столом)
ZONE_SIDE = 0.1              # ... и по 10 % ширины в стороны
ZONE_DOWN = 0.1
REFINE_STEP_SEC = 0.5        # шаг уточнения момента появления назад по времени
REFINE_MAX_SEC = 5.0         # не дальше 5 с до первой детекции на полном кадре
REFINE_MAX_MISS = 2          # остановка после двух кадров подряд без напитка
REFINE_SCORE = 0.5

# --- личная сумка ----------------------------------------------------------
BAG_MIN_SCORE = 0.3
BAG_MIN_DET = 3              # детекций сумки у кассира в одном эпизоде
BAG_PEAK_SCORE = 0.5         # хотя бы одна уверенная
BAG_RUN_GAP_SEC = 1.5        # разрыв внутри эпизода
BAG_MOVE_IOU = 0.7           # рамка «сдвинулась», если IoU с медианной рамкой эпизода < 0.7 ...
BAG_MOVE_FRAC = 0.5          # ... не менее чем у половины детекций эпизода
BAG_INTERIOR_IOU = 0.5       # совпадает с «сумкой», видимой без людей рядом, — предмет интерьера
BAG_GREEN_MAX = 0.2          # доля зелёного в рамке: больше — инкассаторская сумка, не личная вещь
GREEN_HSV = ((35, 50, 60), (90, 255, 255))   # тот же диапазон, что states.GREEN_HSV (инкассаторская сумка)
SIT_WINDOW_SEC = 20.0        # ищем, когда кассир садится, не позже 20 с после эпизода с сумкой
SIT_BASE_SEC = 3.0           # высота рамки стоя — медиана за 3 с до момента
SIT_DROP = 0.85              # начало посадки: рамка ниже 85 % высоты стоя ...
SIT_LOW = 0.72               # ... и в ближайшие 1.5 с опускается ниже 72 %
SIT_HOLD = 0.8               # ... и остаётся ниже 80 % не менее 1.5 с
SIT_HOLD_SEC = 1.5
SIT_INSIDE_FRAC = 0.9       # между сумкой в руках и посадкой кассир внутри кассы ≥ 90 % времени


# ---------------------------------------------------------------------------
# пиксельные признаки по видео (с памятью в results/personal_items/)
# ---------------------------------------------------------------------------

def _stem(path: str) -> str:
    return Path(path).stem.replace("X", "Х")


class PixelMemo:
    """Ленивые пиксельные признаки по видео. Ключи — время и рамка, значения — числа.
    Читает/пишет results/personal_items/<видео>.json только при заданном FEATURES_CACHE."""

    def __init__(self, path: str, info) -> None:
        self.path = path
        self.info = info
        self.data: dict[str, object] = {}
        self.file: Optional[Path] = None
        self.dirty = False
        self._frames: dict[int, Optional[np.ndarray]] = {}
        cache = os.environ.get("FEATURES_CACHE")
        if cache:
            self.file = Path(cache) / "personal_items" / f"{_stem(path)}.json"
            if self.file.exists():
                try:
                    d = json.loads(self.file.read_text(encoding="utf-8"))
                    if d.get("version") == ITEMS_CACHE_VERSION:
                        self.data = dict(d.get("memo") or {})
                except Exception as exc:  # noqa: BLE001
                    log.warning("personal_items cache %s unreadable: %s", self.file, exc)

    def _frame(self, t: float) -> Optional[np.ndarray]:
        k = int(round(t * 100))
        if k not in self._frames:
            try:
                from .video import read_frame_at

                self._frames[k] = read_frame_at(self.path, t, self.info) if os.path.exists(self.path) else None
            except Exception as exc:  # noqa: BLE001
                log.warning("frame %.2f of %s unreadable: %s", t, self.path, exc)
                self._frames[k] = None
        return self._frames[k]

    def green_frac(self, t: float, box) -> Optional[float]:
        """Доля «зелёных» пикселей (инкассаторская сумка) в рамке, сжатой на 15 % с каждой стороны."""
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        key = f"g|{t:.2f}|{x1},{y1},{x2},{y2}"
        if key in self.data:
            v = self.data[key]
            return None if v is None else float(v)
        fr = self._frame(t)
        val: Optional[float] = None
        if fr is not None:
            w, h = x2 - x1, y2 - y1
            a, b = max(0, int(x1 + 0.15 * w)), max(0, int(y1 + 0.15 * h))
            c, d = min(fr.shape[1], int(x2 - 0.15 * w)), min(fr.shape[0], int(y2 - 0.15 * h))
            if c > a and d > b:
                hsv = cv2.cvtColor(fr[b:d, a:c], cv2.COLOR_BGR2HSV)
                val = float((cv2.inRange(hsv, GREEN_HSV[0], GREEN_HSV[1]) > 0).mean())
        self.data[key] = None if val is None else round(val, 4)
        self.dirty = True
        return val

    def crop_detect(self, t: float, roi, classes: tuple[str, ...]) -> Optional[list[tuple[str, float, tuple]]]:
        """Детекции YOLOX на кропе roi (увеличивается до входа детектора) — в координатах кадра."""
        x1, y1, x2, y2 = (int(round(v)) for v in roi)
        key = f"d|{t:.2f}|{x1},{y1},{x2},{y2}"
        if key in self.data:
            v = self.data[key]
            return None if v is None else [(str(c), float(s), tuple(b)) for c, s, b in v]
        fr = self._frame(t)
        out: Optional[list] = None
        if fr is not None:
            try:
                from .detect import get_detector

                crop = fr[max(0, y1):min(fr.shape[0], y2), max(0, x1):min(fr.shape[1], x2)]
                dets = get_detector()(crop, conf=0.3, classes=set(DRINK_CLASSES + FOOD_CLASSES + BAG_CLASSES))
                out = [(d.cls, round(float(d.score), 3),
                        (round(d.xyxy[0] + max(0, x1), 1), round(d.xyxy[1] + max(0, y1), 1),
                         round(d.xyxy[2] + max(0, x1), 1), round(d.xyxy[3] + max(0, y1), 1))) for d in dets]
            except Exception as exc:  # noqa: BLE001 — без детектора уточнение просто не делается
                log.warning("crop detection unavailable: %s", exc)
                out = None
        self.data[key] = out
        self.dirty = True
        if out is None:
            return None
        return [(c, s, b) for c, s, b in out if c in classes]

    def save(self) -> None:
        if not (self.dirty and self.file is not None):
            return
        try:
            self.file.parent.mkdir(parents=True, exist_ok=True)
            self.file.write_text(json.dumps({"version": ITEMS_CACHE_VERSION, "memo": self.data}, ensure_ascii=False),
                                 encoding="utf-8")
            self.dirty = False
        except Exception as exc:  # noqa: BLE001
            log.warning("personal_items cache not written: %s", exc)


# ---------------------------------------------------------------------------
# геометрия
# ---------------------------------------------------------------------------

def _overlap(a, b) -> bool:
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def _iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def desk_zone(desk_bbox, width: int, height: int) -> Optional[tuple[float, float, float, float]]:
    """Зона рабочего стола: столешница + место над ней (предмет в руке, стоящий стакан), доли размеров стола."""
    if not desk_bbox:
        return None
    x1, y1, x2, y2 = desk_bbox
    w, h = x2 - x1, y2 - y1
    return (max(0.0, x1 - ZONE_SIDE * w), max(0.0, y1 - ZONE_UP * h),
            min(float(width), x2 + ZONE_SIDE * w), min(float(height), y2 + ZONE_DOWN * h))


# ---------------------------------------------------------------------------
# сценарий 1: напиток / еда
# ---------------------------------------------------------------------------

def find_drink(perc, zone) -> Optional[dict]:
    """Напиток/еда в зоне стола: ≥ DRINK_MIN_DET детекций, из них ≥ DRINK_MIN_STRONG уверенных."""
    dets = []
    for j, t in enumerate(perc.times):
        for d in perc.objects[j]:
            if d.cls in DRINK_CLASSES + FOOD_CLASSES and d.score >= 0.3 and _overlap(d.xyxy, zone):
                dets.append((float(t), d))
    if len(dets) < DRINK_MIN_DET:
        return None
    strong = [(t, d) for t, d in dets if d.score >= DRINK_STRONG_SCORE]
    if len(strong) < DRINK_MIN_STRONG:
        return None
    best_t, best = max(dets, key=lambda x: x[1].score)
    kind = "food" if sum(1 for _, d in dets if d.cls in FOOD_CLASSES) > len(dets) / 2 else "drink"
    return {"t_first": dets[0][0], "first_box": tuple(dets[0][1].xyxy), "t_best": best_t, "best": best,
            "n": len(dets), "n_strong": len(strong), "kind": kind, "t_last": dets[-1][0]}


def refine_onset(memo: PixelMemo, t_first: float, zone, classes: tuple[str, ...]) -> tuple[float, Optional[tuple]]:
    """Шаг назад от первой детекции на полном кадре: детектор на кропе зоны стола. Возвращает
    (момент появления, рамка в этот момент)."""
    onset, box = t_first, None
    miss = 0
    k = 1
    while k * REFINE_STEP_SEC <= REFINE_MAX_SEC + 1e-6:
        t = t_first - k * REFINE_STEP_SEC
        if t < 0:
            break
        dets = memo.crop_detect(t, zone, classes)
        if dets is None:
            break
        good = [d for d in dets if d[1] >= REFINE_SCORE]
        if good:
            onset, box = t, max(good, key=lambda d: d[1])[2]
            miss = 0
        else:
            miss += 1
            if miss >= REFINE_MAX_MISS:
                break
        k += 1
    return onset, box


# ---------------------------------------------------------------------------
# сценарий 2: личная сумка в руках кассира
# ---------------------------------------------------------------------------

def _bag_runs(perc, roles: dict) -> list[list[tuple[float, object]]]:
    """Эпизоды «сумка в руках кассира» (детекции, сгруппированные по времени), без предметов интерьера."""
    interior = []          # «сумки», которых не касается ни один человек: стул, урна и т. п.
    cand: list[tuple[float, object]] = []
    for j, t in enumerate(perc.times):
        persons = perc.persons[j]
        for d in perc.objects[j]:
            if d.cls not in BAG_CLASSES or d.score < BAG_MIN_SCORE:
                continue
            touching = [p for p in persons if _overlap(p.xyxy, d.xyxy)]
            if not touching:
                interior.append(d.xyxy)
                continue
            if any(roles.get(p.track_id) == "cashier" and p.inside for p in touching):
                cand.append((float(t), d))
    cand = [(t, d) for t, d in cand if not any(_iou(d.xyxy, b) >= BAG_INTERIOR_IOU for b in interior)]
    runs: list[list[tuple[float, object]]] = []
    for t, d in cand:
        if runs and t - runs[-1][-1][0] <= BAG_RUN_GAP_SEC:
            runs[-1].append((t, d))
        else:
            runs.append([(t, d)])
    return runs


def _moving(run) -> bool:
    boxes = np.asarray([d.xyxy for _, d in run], dtype=float)
    med = np.median(boxes, axis=0)
    moved = sum(1 for b in boxes if _iou(b, med) < BAG_MOVE_IOU)
    return moved >= BAG_MOVE_FRAC * len(boxes)


def find_bag(perc, roles: dict, memo: PixelMemo) -> Optional[dict]:
    for run in _bag_runs(perc, roles):
        if len(run) < BAG_MIN_DET or max(d.score for _, d in run) < BAG_PEAK_SCORE:
            continue
        if not _moving(run):
            continue
        top = sorted(run, key=lambda x: -x[1].score)[:3]
        greens = [memo.green_frac(t, d.xyxy) for t, d in top]
        greens = [g for g in greens if g is not None]
        if not greens or float(np.median(greens)) > BAG_GREEN_MAX:
            continue      # инкассаторская (зелёная) сумка или нет кадров для проверки — не личная вещь
        best_t, best = top[0]
        return {"t_first": run[0][0], "t_last": run[-1][0], "t_best": best_t, "best": best, "n": len(run),
                "green": float(np.median(greens))}
    return None


def sit_down_after(sc, t_from: float, t_to: float) -> Optional[float]:
    """Момент, когда кассир садится за рабочее место: высота его рамки резко падает относительно высоты
    стоя за предыдущие SIT_BASE_SEC и остаётся низкой ≥ SIT_HOLD_SEC (кассир у стола)."""
    from .scene import GRID_FPS

    h = np.asarray(sc.cashier_h, dtype=float)
    n = len(h)
    nb = max(2, int(SIT_BASE_SEC * GRID_FPS))
    nh = max(1, int(SIT_HOLD_SEC * GRID_FPS))
    i0 = int(np.searchsorted(sc.t, t_from))
    i1 = int(np.searchsorted(sc.t, t_to))
    for k in range(max(i0, 1), min(i1, n - nh)):
        if h[k] <= 0 or not sc.cashier_at_desk[k]:
            continue
        pre = h[max(0, k - nb):k]
        pre = pre[pre > 0]
        if len(pre) < 2:
            continue
        base = float(np.median(pre))
        if h[k] > SIT_DROP * base:
            continue
        nxt = h[k:k + nh + 1]
        nxt = nxt[nxt > 0]
        if len(nxt) < nh or nxt.min() > SIT_LOW * base or (nxt > SIT_HOLD * base).any():
            continue
        return float(sc.t[k])
    return None


# ---------------------------------------------------------------------------
# правило
# ---------------------------------------------------------------------------

def _clip(info, t: float) -> float:
    return float(max(0.0, min(t, info.duration_sec - 0.1)))


def rule_t8(ctx: VideoContext) -> list[Event]:
    """T8: напиток/еда в зоне стола кассира или личная сумка, внесённая кассиром в кассу."""
    sc = getattr(ctx, "scene", None)
    perc = (ctx.cache or {}).get("perception")
    if sc is None or perc is None or len(sc.t) == 0:
        return []
    desk = (sc.anchors or {}).get("desk")
    zone = desk_zone(desk["bbox"] if desk else None, sc.info.width, sc.info.height)
    memo = PixelMemo(ctx.path, ctx.info)
    events: list[Event] = []
    try:
        if zone is not None:
            dr = find_drink(perc, zone)
            if dr is not None:
                classes = FOOD_CLASSES if dr["kind"] == "food" else DRINK_CLASSES
                onset, obox = refine_onset(memo, dr["t_first"], zone, classes)
                what = "еда" if dr["kind"] == "food" else "стакан/бутылка с напитком"
                best = dr["best"]
                desc = (
                    f"С {onset:.1f} с в зоне рабочего стола кассира находится {what}: предмет распознан "
                    f"{dr['n']} раз до {dr['t_last']:.1f} с и находится на рабочем месте кассира, где ведутся операции с "
                    f"наличными. Личные вещи, еда и напитки работников хранятся вне кассового узла."
                )
                conf = min(0.85, 0.66 + 0.02 * dr["n_strong"])
                boxes = [obox or dr["first_box"], dr["first_box"], tuple(best.xyxy)]
                events.append(Event(
                    type_code="T8", timecode_sec=round(onset, 2), end_sec=round(dr["t_last"], 2),
                    confidence=round(conf, 2), description=desc,
                    evidence_times=[_clip(ctx.info, onset), _clip(ctx.info, dr["t_first"]), _clip(ctx.info, dr["t_best"])],
                    evidence_captions=[f"{'еда' if dr['kind'] == 'food' else 'напиток'} появляется на рабочем месте",
                                       "напиток/еда в зоне стола кассира", "напиток/еда у кассира на рабочем месте"],
                    extras={"evidence_boxes": [tuple(int(v) for v in b) for b in boxes], "kind": dr["kind"],
                            "n_det": dr["n"], "t_first_full": dr["t_first"]},
                ))
        bag = find_bag(perc, sc.roles, memo)
        if bag is not None:
            t_sit = sit_down_after(sc, bag["t_last"], bag["t_last"] + SIT_WINDOW_SEC)
            if t_sit is not None:
                # между эпизодом с сумкой и посадкой кассир не выходил из кассы (сумку не уносил)
                i0, i1 = int(np.searchsorted(sc.t, bag["t_last"])), int(np.searchsorted(sc.t, t_sit)) + 1
                seg = np.asarray(sc.cashier_inside[i0:i1], bool)
                if len(seg) and seg.mean() < SIT_INSIDE_FRAC:
                    t_sit = None
            t0 = t_sit if t_sit is not None else bag["t_first"]
            best = bag["best"]
            tail = (f"не выходя из кассового помещения, в {t_sit:.1f} с кассир с внесённой сумкой занимает рабочее "
                    f"место (садится за стол)" if t_sit is not None else "кассир находится с ней в кассовом помещении")
            desc = (
                f"С {bag['t_first']:.1f} по {bag['t_last']:.1f} с у кассира в руках сумка (личная вещь, не "
                f"инкассаторская сумка), {tail}. Личные вещи работников должны храниться вне кассового узла."
            )
            conf = min(0.8, 0.6 + 0.2 * float(best.score))
            ev_t = [_clip(ctx.info, bag["t_best"]), _clip(ctx.info, t0), _clip(ctx.info, t0 + 2.0)]
            events.append(Event(
                type_code="T8", timecode_sec=round(t0, 2), end_sec=None, confidence=round(conf, 2), description=desc,
                evidence_times=ev_t,
                evidence_captions=["кассир вносит личную сумку", "кассир занимает рабочее место с личными вещами",
                                   "личные вещи в кассовом помещении"],
                extras={"evidence_boxes": [tuple(int(v) for v in best.xyxy), None, None], "kind": "bag",
                        "n_det": bag["n"], "green": bag["green"], "t_sit": t_sit},
            ))
    finally:
        memo.save()
    events.sort(key=lambda e: e.timecode_sec)
    return events
