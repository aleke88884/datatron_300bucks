"""Признаки состояний сцены по кадрам (2 к/с): дверь, сейф, экран ноутбука, столешница и предметы на ней.

Только числовые ряды, без изображений — их можно хранить в репозитории (кадры датасета публиковать нельзя).
Эти ряды собираются scripts/collect_features.py на всех видео; по ним подбираются пороги правил T1–T13.

Якоря сцены строятся по медианному кадру видео (люди «вымываются» медианой):
  - столешница — по цвету дерева (HSV), выпуклая оболочка и ось (PCA);
  - сейф — светло-серый прямоугольник в правой части кадра;
  - ноутбук — медиана детекций класса laptop;
  - дверь — полоса матового стекла в дверном проёме (нормированные координаты, калибровка под эту камеру).
"""
from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from .detect import Detection, get_detector
from .perception import Perception, PersonObs
from .video import VideoInfo, iter_frames

log = logging.getLogger("pipeline.states")

STATE_FPS = 2.0
WOOD_HSV = ((8, 40, 120), (28, 140, 235))      # светлое дерево столешницы
GREEN_HSV = ((35, 50, 60), (90, 255, 255))      # зелёная инкассаторская сумка
DOOR_BAND = (0.237, 0.242, 0.431, 0.336)        # матовая полоса на стекле закрытой двери (доли ширины/высоты)
SAFE_GREY = ((0, 0, 95), (180, 35, 170))        # корпус сейфа: светло-серый, малонасыщенный
ITEM_CLASSES = {"handbag", "backpack", "suitcase", "cup", "bottle", "wine glass", "bowl", "cell phone", "book", "remote"}
DESK_UP = 70          # предметы стоят НА столе: расширяем область стола вверх (пикс. при высоте 848)
DIFF_THR = 40         # порог изменения относительно медианного кадра (сумма |dB|+|dG|+|dR|)
BLOB_MIN_AREA = 120


def _box_overlaps(box, persons: list[PersonObs], margin: float = 0.0) -> bool:
    x1, y1, x2, y2 = box
    for p in persons:
        a, b, c, d = p.xyxy
        if a - margin < x2 and c + margin > x1 and b - margin < y2 and d + margin > y1:
            return True
    return False


def median_frame(path: str, info: VideoInfo, max_frames: int = 25) -> np.ndarray:
    every = max(1, int(round(info.frame_count / max_frames)))
    frames = [f for _, _, f in iter_frames(path, every_n=every)]
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


def grey_box(median: np.ndarray) -> Optional[dict]:
    """Корпус сейфа: светло-серый заполненный прямоугольник в правых 60 % кадра, средняя высота."""
    H, W = median.shape[:2]
    hsv = cv2.cvtColor(median, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, SAFE_GREY[0], SAFE_GREY[1])
    m[: int(0.27 * H)] = 0
    m[int(0.62 * H):] = 0
    m[:, : int(0.4 * W)] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    best = None
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < 1500 or w < 40 or h < 50 or a / (w * h) < 0.6:
            continue
        if best is None or a > best[4]:
            best = st[i]
    if best is None:
        return None
    x, y, w, h, a = (int(v) for v in best)
    return {"bbox": [x, y, x + w, y + h], "area": a, "fill": float(a / (w * h))}


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


def _desk_crop_box(desk: dict, H: int, W: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = desk["bbox"]
    s = H / 848.0
    return max(0, x1 - int(20 * s)), max(0, y1 - int(DESK_UP * s)), min(W, x2 + int(20 * s)), min(H, y2 + int(10 * s))


def extract_states(path: str, info: VideoInfo, perc: Perception, progress=None, detect_items: bool = True) -> dict:
    """Якоря сцены + признаки по кадрам с частотой STATE_FPS."""
    H, W = info.height, info.width
    med = median_frame(path, info)
    med_gray = cv2.cvtColor(med, cv2.COLOR_BGR2GRAY)
    desk = desk_region(med)
    safe = grey_box(med)
    lap = laptop_box(perc)
    s = H / 848.0
    door = (int(DOOR_BAND[0] * W), int(DOOR_BAND[1] * H), int(DOOR_BAND[2] * W), int(DOOR_BAND[3] * H))
    desk_mask = None
    desk_crop = None
    if desk is not None:
        desk_mask = np.zeros((H, W), np.uint8)
        cv2.fillConvexPoly(desk_mask, np.asarray(desk["hull"], np.int32), 255)
        # предметы стоят на столе: добавляем полосу над оболочкой
        desk_mask = cv2.dilate(desk_mask, np.ones((int(DESK_UP * s) * 2 + 1, 3), np.uint8), anchor=(1, int(DESK_UP * s) * 2))
        desk_crop = _desk_crop_box(desk, H, W)
    det = get_detector() if (detect_items and desk is not None) else None
    step = max(1, int(round(info.fps / STATE_FPS)))
    rows: list[dict] = []
    total = max(1, info.frame_count)
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
            sx1, sy1, sx2, sy2 = safe["bbox"]
            sw = sx2 - sx1
            face = gray[sy1:sy2, sx1:sx2]
            left = gray[sy1:sy2, max(0, sx1 - int(0.6 * sw)):sx1]
            edges = cv2.Canny(face, 50, 120)
            diff = np.abs(face.astype(int) - med_gray[sy1:sy2, sx1:sx2].astype(int)).mean()
            row["safe"] = {
                "face_mean": float(face.mean()), "face_std": float(face.std()), "edges": float((edges > 0).mean()),
                "left_mean": float(left.mean()) if left.size else None, "left_std": float(left.std()) if left.size else None,
                "diff_vs_median": float(diff),
                "occluded": _box_overlaps((sx1 - int(0.6 * sw), sy1, sx2, sy2), persons, 5 * s),
            }
        # --- экран ноутбука
        if lap is not None:
            lx1, ly1, lx2, ly2 = _screen_region(lap["bbox"])
            scr = frame[ly1:ly2, lx1:lx2]
            g = gray[ly1:ly2, lx1:lx2]
            row["screen"] = {
                "mean": float(g.mean()), "std": float(g.std()), "p90": float(np.percentile(g, 90)) if g.size else None,
                "blue_minus_red": float(scr[..., 0].mean() - scr[..., 2].mean()) if scr.size else None,
                "occluded": _box_overlaps((lx1, ly1, lx2, ly2), persons, 0),
            }
        # --- столешница: доля дерева, изменившиеся пятна относительно медианы, зелёная сумка, предметы
        if desk is not None:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            wood = cv2.inRange(hsv, WOOD_HSV[0], WOOD_HSV[1])
            hull_mask = np.zeros((H, W), np.uint8)
            cv2.fillConvexPoly(hull_mask, np.asarray(desk["hull"], np.int32), 255)
            hull_area = max(1, int((hull_mask > 0).sum()))
            wood_frac = float(((wood > 0) & (hull_mask > 0)).sum() / hull_area)
            diff = np.abs(frame.astype(int) - med.astype(int)).sum(2)
            changed = ((diff > DIFF_THR) & (desk_mask > 0)).astype(np.uint8)
            changed = cv2.morphologyEx(changed, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            n, lab, st, _ = cv2.connectedComponentsWithStats(changed)
            blobs = []
            for i in range(1, n):
                bx, by, bw, bh, ba = (int(v) for v in st[i])
                if ba < BLOB_MIN_AREA * s * s:
                    continue
                sel = lab == i
                hmean = hsv[sel].mean(0)
                blobs.append({
                    "bbox": [bx, by, bx + bw, by + bh], "area": ba,
                    "hsv": [float(hmean[0]), float(hmean[1]), float(hmean[2])],
                    "person": _box_overlaps((bx, by, bx + bw, by + bh), persons, 0),
                })
            blobs.sort(key=lambda b: -b["area"])
            green = cv2.inRange(hsv, GREEN_HSV[0], GREEN_HSV[1])
            green = ((green > 0) & (desk_mask > 0)).astype(np.uint8)
            gn, glab, gst, _ = cv2.connectedComponentsWithStats(green)
            gbest = max(((int(gst[i, 4]), i) for i in range(1, gn)), default=(0, 0))
            gbox = [int(v) for v in gst[gbest[1]][:4]] if gbest[0] > 0 else None
            row["desk"] = {
                "wood_frac": wood_frac,
                "blobs": blobs[:6],
                "green_area": gbest[0],
                "green_bbox": [gbox[0], gbox[1], gbox[0] + gbox[2], gbox[1] + gbox[3]] if gbox else None,
                "person_over_desk": _box_overlaps(tuple(desk["bbox"]), persons, 0),
            }
            if det is not None and desk_crop is not None:
                cx1, cy1, cx2, cy2 = desk_crop
                crop = frame[cy1:cy2, cx1:cx2]
                up = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                items = []
                for d in det(up, conf=0.2):
                    if d.cls not in ITEM_CLASSES:
                        continue
                    bx1, by1, bx2, by2 = d.xyxy[0] / 2 + cx1, d.xyxy[1] / 2 + cy1, d.xyxy[2] / 2 + cx1, d.xyxy[3] / 2 + cy1
                    ix1, iy1, ix2, iy2 = int(bx1), int(by1), max(int(bx1) + 1, int(bx2)), max(int(by1) + 1, int(by2))
                    static = float(np.abs(gray[iy1:iy2, ix1:ix2].astype(int) - med_gray[iy1:iy2, ix1:ix2].astype(int)).mean())
                    items.append({"cls": d.cls, "score": round(d.score, 3), "bbox": [round(bx1, 1), round(by1, 1), round(bx2, 1), round(by2, 1)],
                                  "diff_vs_median": round(static, 1), "person": _box_overlaps((bx1, by1, bx2, by2), persons, 0)})
                row["items"] = items
        rows.append(row)
        if progress and idx % (step * 30) == 0:
            progress("states", 55 + int(25 * idx / total))
    scene = {
        "width": W, "height": H, "fps": info.fps, "duration_sec": info.duration_sec,
        "desk": desk, "safe": safe, "laptop": lap, "door_band": list(door),
        "median_brightness": float(med_gray.mean()),
    }
    return {"scene": scene, "state_fps": STATE_FPS, "frames": rows}
