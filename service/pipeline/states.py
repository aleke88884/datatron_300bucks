"""Признаки состояний сцены по кадрам (2 к/с): дверь, сейф, экран ноутбука, столешница и предметы на ней.

Только числовые ряды, без изображений — их можно хранить в репозитории (кадры датасета публиковать нельзя).
Эти ряды собираются scripts/collect_features.py на всех видео; по ним подбираются пороги правил T1–T13
(scripts/eval_rules.py), а в проде те же функции вызывает analyze_video.

Якоря сцены:
  - столешница — по цвету дерева (HSV) на медианном кадре: выпуклая оболочка и ось (PCA);
  - сейф — светло-серый заполненный прямоугольник рядом со столом; ищется на нескольких кадрах и берётся
    самый крупный устойчивый кандидат (в закрытом состоянии фронт сейфа — самый большой однотонный блок);
  - ноутбук — медиана детекций класса laptop;
  - дверь — полоса матового стекла в дверном проёме (нормированные координаты, калибровка под эту камеру).

Признаки по кадрам:
  - door: яркость/текстура матовой полосы; occluded — в проёме человек;
  - safe: dark_face / dark_left — доля тёмных пикселей на фронте и слева от него (открытая дверца
    обнажает тёмный проём и смещает светлую панель), face_std, diff_vs_median; occluded — человек у сейфа;
  - screen: mean/std/p90/blue_minus_red по верхней части рамки ноутбука (активен / заблокирован / выключен);
  - desk: wood_frac, green_area (инкассаторская сумка), static_objs — статичные предметы на столешнице:
    пиксели, которые когда-либо были «деревом» (wood_ever), но сейчас не дерево и не перекрыты человеком,
    устойчивые ≥ 1.5 с (деньги, карточки, стакан, сумка, положенные на стол).
"""
from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from .perception import Perception, PersonObs
from .video import VideoInfo, iter_frames

log = logging.getLogger("pipeline.states")

STATE_FPS = 2.0
WOOD_HSV = ((8, 40, 120), (28, 140, 235))      # светлое дерево столешницы
GREEN_HSV = ((35, 50, 60), (90, 255, 255))      # зелёная инкассаторская сумка
DOOR_BAND = (0.237, 0.242, 0.431, 0.336)        # матовая полоса на стекле закрытой двери (доли ширины/высоты)
SAFE_GREY = ((0, 0, 95), (180, 35, 170))        # корпус сейфа: светло-серый, малонасыщенный
DESK_UP = 70          # предметы стоят НА столе: расширяем область стола вверх (пикс. при высоте 848)
OBJ_MIN_AREA = 100    # минимальная площадь предмета на столе (пикс. при 464x848)
OBJ_STATIC_FRAMES = 3 # предмет «статичен», если наблюдается ≥ 3 кадров подряд (1.5 с при 2 к/с)
OBJ_MATCH_IOU = 0.4
PERSON_SHRINK = 0.2   # рамка человека сжимается на 20 % с каждой стороны для теста перекрытия предметов
DARK_THR = 70         # «тёмный» пиксель внутри/около сейфа
SAFE_LEFT_EXT = 0.8   # ширина зоны слева от сейфа (доля ширины сейфа): туда уходит дверца / виден проём


def _box_overlaps(box, persons: list[PersonObs], margin: float = 0.0, shrink: float = 0.0) -> bool:
    x1, y1, x2, y2 = box
    for p in persons:
        a, b, c, d = p.xyxy
        if shrink:
            w, h = (c - a) * shrink, (d - b) * shrink
            a, b, c, d = a + w, b + h, c - w, d - h
        if a - margin < x2 and c + margin > x1 and b - margin < y2 and d + margin > y1:
            return True
    return False


def _iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def sample_frames(path: str, info: VideoInfo, max_frames: int = 25) -> list[np.ndarray]:
    every = max(1, int(round(info.frame_count / max_frames)))
    return [f for _, _, f in iter_frames(path, every_n=every)][:max_frames]


def median_frame(path: str, info: VideoInfo, max_frames: int = 25, frames: Optional[list[np.ndarray]] = None) -> np.ndarray:
    frames = frames if frames is not None else sample_frames(path, info, max_frames)
    if not frames:
        raise RuntimeError("no frames for median")
    return np.median(np.stack(frames[:max_frames]), axis=0).astype(np.uint8)


def wood_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, WOOD_HSV[0], WOOD_HSV[1])
    m[: int(0.18 * bgr.shape[0])] = 0     # сверху стол быть не может (там стекло/плакаты)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    return m


def desk_region(median: np.ndarray) -> Optional[dict]:
    """Столешница: объединение крупных деревянных компонент → bbox, выпуклая оболочка, ось."""
    m = wood_mask(median)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    keep = [i for i in range(1, n) if st[i, 4] >= 500]
    if not keep:
        return None
    sel = np.isin(lab, keep).astype(np.uint8)
    pts = cv2.findNonZero(sel)
    hull = cv2.convexHull(pts).reshape(-1, 2)
    x, y, w, h = cv2.boundingRect(pts)
    mean = pts.reshape(-1, 2).astype(np.float32).mean(0)
    cov = np.cov((pts.reshape(-1, 2).astype(np.float32) - mean).T)
    evals, evecs = np.linalg.eigh(cov)
    axis = evecs[:, int(np.argmax(evals))]
    return {
        "bbox": [int(x), int(y), int(x + w), int(y + h)],
        "hull": [[int(a), int(b)] for a, b in hull],
        "area": int(sel.sum()),
        "center": [float(mean[0]), float(mean[1])],
        "axis": [float(axis[0]), float(axis[1])],
    }


def _grey_candidates(frame: np.ndarray) -> list[dict]:
    H, W = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, SAFE_GREY[0], SAFE_GREY[1])
    m[: int(0.25 * H)] = 0
    m[int(0.66 * H):] = 0
    m[:, : int(0.35 * W)] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    out = []
    for i in range(1, n):
        x, y, w, h, a = (int(v) for v in st[i])
        if a < 900 or w < 40 or h < 45 or a / (w * h) < 0.5:
            continue
        out.append({"bbox": [x, y, x + w, y + h], "area": a, "fill": float(a / (w * h))})
    return out


def grey_box(median: np.ndarray) -> Optional[dict]:
    """Совместимость: лучший серый кандидат на одном кадре."""
    c = _grey_candidates(median)
    return max(c, key=lambda d: d["area"]) if c else None


def safe_anchor(frames: list[np.ndarray], desk: Optional[dict]) -> Optional[dict]:
    """Сейф по нескольким кадрам: кандидаты кластеризуются по IoU, выбирается устойчивый кластер
    (виден ≥ 30 % кадров) с наибольшей медианной площадью рядом со столом; bbox — самый крупный член
    кластера (закрытое состояние). Полосы у правого края кадра (стена) отбрасываются, если узкие."""
    clusters: list[dict] = []
    for fr in frames:
        W = fr.shape[1]
        for c in _grey_candidates(fr):
            x1, y1, x2, y2 = c["bbox"]
            if x2 >= W - 2 and (x2 - x1) < 60:      # узкая полоса у правого края — не сейф
                continue
            for cl in clusters:
                if _iou(cl["bbox"], c["bbox"]) >= 0.4:
                    cl["members"].append(c)
                    if c["area"] > cl["area"]:
                        cl["bbox"], cl["area"], cl["fill"] = c["bbox"], c["area"], c["fill"]
                    break
            else:
                clusters.append({"bbox": c["bbox"], "area": c["area"], "fill": c["fill"], "members": [c]})
    if not clusters:
        return None
    nfr = max(1, len(frames))
    good = [cl for cl in clusters if len(cl["members"]) >= max(2, 0.3 * nfr)]
    if not good:
        good = clusters
    if desk is not None:
        dx1, dy1, dx2, dy2 = desk["bbox"]
        dw = dx2 - dx1

        def near(cl):
            x1, y1, x2, y2 = cl["bbox"]
            gap_x = max(0, x1 - dx2, dx1 - x2)
            gap_y = max(0, y1 - dy2, dy1 - y2)
            return gap_x <= 0.6 * dw and gap_y <= 0.8 * (dy2 - dy1) + 60
        near_cl = [cl for cl in good if near(cl)]
        if near_cl:
            good = near_cl
    best = max(good, key=lambda cl: float(np.median([m["area"] for m in cl["members"]])))
    return {"bbox": best["bbox"], "area": best["area"], "fill": best["fill"], "seen": len(best["members"]), "frames": nfr}


def laptop_box(perc: Perception, min_score: float = 0.6) -> Optional[dict]:
    boxes = [d.xyxy for ds in perc.objects for d in ds if d.cls == "laptop" and d.score >= min_score]
    if len(boxes) < 3:
        return None
    arr = np.asarray(boxes)
    med = np.median(arr, axis=0)
    return {"bbox": [float(v) for v in med], "n": int(len(boxes))}


def _screen_region(lb: list[float]) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = lb
    return int(x1), int(y1), int(x2), int(y1 + 0.6 * (y2 - y1))


def extract_states(path: str, info: VideoInfo, perc: Perception, progress=None, detect_items: bool = False) -> dict:
    """Якоря сцены + признаки по кадрам с частотой STATE_FPS. detect_items оставлен для совместимости (не используется)."""
    H, W = info.height, info.width
    s = H / 848.0
    samples = sample_frames(path, info, 25)
    med = median_frame(path, info, frames=samples)
    med_gray = cv2.cvtColor(med, cv2.COLOR_BGR2GRAY)
    desk = desk_region(med)
    safe = safe_anchor(samples[::2] if len(samples) > 12 else samples, desk)
    lap = laptop_box(perc)
    door = (int(DOOR_BAND[0] * W), int(DOOR_BAND[1] * H), int(DOOR_BAND[2] * W), int(DOOR_BAND[3] * H))

    hull_mask = None
    desk_mask = None
    if desk is not None:
        hull_mask = np.zeros((H, W), np.uint8)
        cv2.fillConvexPoly(hull_mask, np.asarray(desk["hull"], np.int32), 255)
        desk_mask = cv2.dilate(hull_mask, np.ones((int(DESK_UP * s) * 2 + 1, 3), np.uint8), anchor=(1, int(DESK_UP * s) * 2))
        # wood_ever: пиксели столешницы, которые хоть раз были деревом (по выборке кадров + медиане)
        wood_ever = (wood_mask(med) > 0) & (hull_mask > 0)
        for fr in samples:
            wood_ever |= (wood_mask(fr) > 0) & (hull_mask > 0)
        wood_ever = cv2.morphologyEx(wood_ever.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)) > 0
    step = max(1, int(round(info.fps / STATE_FPS)))
    rows: list[dict] = []
    total = max(1, info.frame_count)
    prev_objs: list[dict] = []
    sx1 = sy1 = sx2 = sy2 = 0
    if safe is not None:
        sx1, sy1, sx2, sy2 = safe["bbox"]
        sw = sx2 - sx1
        lx1 = max(0, sx1 - int(SAFE_LEFT_EXT * sw))
    for idx, t, frame in iter_frames(path, every_n=step):
        persons = perc.persons[perc.frame_index(t)] if perc.persons else []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        row: dict = {"t": round(t, 2), "brightness": float(gray.mean()), "n_inside": sum(1 for p in persons if p.inside)}
        # --- дверь: матовая полоса в проёме
        x1, y1, x2, y2 = door
        band = gray[y1:y2, x1:x2]
        row["door"] = {"mean": float(band.mean()), "std": float(band.std()), "occluded": _box_overlaps(door, persons, 10 * s)}
        # --- сейф
        if safe is not None:
            face = gray[sy1:sy2, sx1:sx2]
            left = gray[sy1:sy2, lx1:sx1]
            diff = np.abs(face.astype(int) - med_gray[sy1:sy2, sx1:sx2].astype(int)).mean()
            row["safe"] = {
                "face_mean": float(face.mean()), "face_std": float(face.std()),
                "dark_face": float((face < DARK_THR).mean()) if face.size else None,
                "dark_left": float((left < DARK_THR).mean()) if left.size else None,
                "diff_vs_median": float(diff),
                "occluded": _box_overlaps((sx1, sy1, sx2, sy2), persons, 0, shrink=PERSON_SHRINK),
            }
        # --- экран ноутбука
        if lap is not None:
            lx1_, ly1_, lx2_, ly2_ = _screen_region(lap["bbox"])
            scr = frame[ly1_:ly2_, lx1_:lx2_]
            g = gray[ly1_:ly2_, lx1_:lx2_]
            row["screen"] = {
                "mean": float(g.mean()) if g.size else None, "std": float(g.std()) if g.size else None,
                "p90": float(np.percentile(g, 90)) if g.size else None,
                "blue_minus_red": float(scr[..., 0].mean() - scr[..., 2].mean()) if scr.size else None,
                "occluded": _box_overlaps((lx1_, ly1_, lx2_, ly2_), persons, 0, shrink=PERSON_SHRINK),
            }
        # --- столешница
        if desk is not None:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            wood = cv2.inRange(hsv, WOOD_HSV[0], WOOD_HSV[1]) > 0
            hull_area = max(1, int((hull_mask > 0).sum()))
            wood_frac = float((wood & (hull_mask > 0)).sum() / hull_area)
            # статичные предметы: было дерево, сейчас не дерево; тени (древесный оттенок, но темнее) — не предметы
            shadow = (hsv[..., 0] >= WOOD_HSV[0][0]) & (hsv[..., 0] <= WOOD_HSV[1][0]) & (hsv[..., 2] < WOOD_HSV[0][2])
            bright = hsv[..., 2] >= 232          # пересвеченные светлые купюры/бумага: ярче любого дерева
            obj = (wood_ever & (~wood | bright) & ~shadow).astype(np.uint8)
            obj = cv2.morphologyEx(obj, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            obj = cv2.morphologyEx(obj, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
            n, lab, st, _ = cv2.connectedComponentsWithStats(obj)
            cur: list[dict] = []
            for i in range(1, n):
                bx, by, bw, bh, ba = (int(v) for v in st[i])
                if ba < OBJ_MIN_AREA * s * s:
                    continue
                sel = lab == i
                hmean = hsv[sel].mean(0)
                box = (bx, by, bx + bw, by + bh)
                age = 1
                for po in prev_objs:
                    if _iou(po["bbox"], box) >= OBJ_MATCH_IOU:
                        age = po["age"] + 1
                        break
                cur.append({
                    "bbox": list(box), "area": ba,
                    "hsv": [float(hmean[0]), float(hmean[1]), float(hmean[2])],
                    "person": _box_overlaps(box, persons, 0, shrink=PERSON_SHRINK),
                    "age": age,
                })
            cur.sort(key=lambda b: -b["area"])
            prev_objs = cur
            static_objs = [b for b in cur if b["age"] >= OBJ_STATIC_FRAMES]
            green = cv2.inRange(hsv, GREEN_HSV[0], GREEN_HSV[1])
            green = ((green > 0) & (desk_mask > 0)).astype(np.uint8)
            gn, glab, gst, _ = cv2.connectedComponentsWithStats(green)
            gbest = max(((int(gst[i, 4]), i) for i in range(1, gn)), default=(0, 0))
            gbox = [int(v) for v in gst[gbest[1]][:4]] if gbest[0] > 0 else None
            row["desk"] = {
                "wood_frac": wood_frac,
                "objs": cur[:8],
                "static_objs": static_objs[:8],
                "green_area": gbest[0],
                "green_bbox": [gbox[0], gbox[1], gbox[0] + gbox[2], gbox[1] + gbox[3]] if gbox else None,
                "person_over_desk": _box_overlaps(tuple(desk["bbox"]), persons, 0),
            }
        rows.append(row)
        if progress and idx % (step * 10) == 0:
            progress("states", int(100 * idx / total))
    scene = {
        "width": W, "height": H, "fps": info.fps, "duration_sec": info.duration_sec,
        "desk": desk, "safe": safe, "laptop": lap, "door_band": list(door),
        "median_brightness": float(med_gray.mean()),
        "version": 2,
    }
    return {"scene": scene, "state_fps": STATE_FPS, "frames": rows}
