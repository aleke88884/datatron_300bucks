"""Поза людей (17 ключевых точек COCO) поверх уже найденных рамок людей — слой для правил по рукам (T6, T11, T12).

Модель: RTMPose-m (body7, 256x192, ONNX; код и веса OpenMMLab MMPose, Apache-2.0 — см. docs/лицензии, оговорка
про датасеты обучения в README). Модель top-down: детектор людей не нужен — используются рамки YOLOX из
perception (те же 3 к/с), поэтому дополнительная стоимость — только кроп + один прогон сети на человека.
Лица не распознаются и не сохраняются: на выходе только координаты суставов.

Результат (dict, сериализуется в results/hands/<видео>.json, только числа):

    {"version": 1, "model": "rtmpose-m_body7_256x192",
     "times": [t0, t1, ...],                                    # моменты кадров (как в perception)
     "poses": [[{"track_id": id, "kpts": [[x, y, score] * 17]}, ...], ...]}

Индексы ключевых точек (COCO-17): 5/6 плечи, 7/8 локти, 9/10 кисти, 11/12 бёдра.
Признаки, производные от позы (pose_series), — только относительные: координаты кисти в системе тела
(начало — середина плеч, единица — ширина плеч/длина торса), скорость кисти в ширинах плеч в секунду.
Никаких пиксельных порогов, привязанных к камере.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("pipeline.pose")

POSE_VERSION = 1
POSE_MODEL_NAME = "rtmpose-m_body7_256x192"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DEFAULT_MODEL = MODELS_DIR / "rtmpose_m_body7.onnx"
INPUT_W, INPUT_H = 192, 256
SIMCC_SPLIT = 2.0
BOX_PADDING = 1.25                 # как в конфиге RTMPose (TopDownGetBboxCenterScale)
MEAN = np.array([123.675, 116.28, 103.53], np.float32)
STD = np.array([58.395, 57.12, 57.375], np.float32)
KP_MIN_SCORE = 0.3                 # ниже — точка «не видна» (неизвестно, а не «нарушение»)

L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP = 5, 6, 7, 8, 9, 10, 11, 12


def model_path() -> Path:
    return Path(os.environ.get("POSE_MODEL") or DEFAULT_MODEL)


def pose_enabled() -> bool:
    return os.environ.get("POSE", "1") != "0"


class PoseEstimator:
    """RTMPose top-down через onnxruntime (CPU). Потокобезопасная ленивая загрузка."""

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path or model_path())
        self._sess = None
        self._lock = threading.Lock()

    def available(self) -> bool:
        return self.path.exists()

    def _session(self):
        if self._sess is None:
            with self._lock:
                if self._sess is None:
                    import onnxruntime as ort

                    so = ort.SessionOptions()
                    so.intra_op_num_threads = max(1, min(4, (os.cpu_count() or 2)))
                    self._sess = ort.InferenceSession(str(self.path), so, providers=["CPUExecutionProvider"])
        return self._sess

    @staticmethod
    def _box_transform(xyxy) -> tuple[float, float, float, float]:
        """Центр и размер области кропа: рамка × 1.25, приведённая к пропорции входа 192:256."""
        x1, y1, x2, y2 = (float(v) for v in xyxy)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        w, h = max(x2 - x1, 1.0) * BOX_PADDING, max(y2 - y1, 1.0) * BOX_PADDING
        aspect = INPUT_W / INPUT_H
        if w > h * aspect:
            h = w / aspect
        else:
            w = h * aspect
        return cx, cy, w, h

    def __call__(self, frame_bgr: np.ndarray, boxes: list) -> list[np.ndarray]:
        """Для каждой рамки — массив (17, 3): x, y (пиксели кадра), score."""
        if not boxes:
            return []
        import cv2

        batch = []
        trs = []
        for b in boxes:
            cx, cy, w, h = self._box_transform(b)
            sx, sy = INPUT_W / w, INPUT_H / h
            m = np.array([[sx, 0.0, -(cx - w / 2.0) * sx], [0.0, sy, -(cy - h / 2.0) * sy]], np.float32)
            crop = cv2.warpAffine(frame_bgr, m, (INPUT_W, INPUT_H), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
            img = (crop[:, :, ::-1].astype(np.float32) - MEAN) / STD
            batch.append(img.transpose(2, 0, 1))
            trs.append((cx, cy, w, h))
        sess = self._session()
        sx_out, sy_out = sess.run(None, {sess.get_inputs()[0].name: np.stack(batch).astype(np.float32)})
        out = []
        for k, (cx, cy, w, h) in enumerate(trs):
            ax = sx_out[k]          # (17, 384)
            ay = sy_out[k]          # (17, 512)
            lx = ax.argmax(axis=1).astype(np.float32) / SIMCC_SPLIT
            ly = ay.argmax(axis=1).astype(np.float32) / SIMCC_SPLIT
            score = np.minimum(ax.max(axis=1), ay.max(axis=1))
            x = lx * w / INPUT_W + cx - w / 2.0
            y = ly * h / INPUT_H + cy - h / 2.0
            out.append(np.stack([x, y, score], axis=1))
        return out


_ESTIMATOR: Optional[PoseEstimator] = None


def get_estimator() -> PoseEstimator:
    global _ESTIMATOR
    if _ESTIMATOR is None:
        _ESTIMATOR = PoseEstimator()
    return _ESTIMATOR


def _round_kpts(k: np.ndarray) -> list:
    """Компактная запись для кеша: координаты — 0.1 пикс., уверенность — 0.01."""
    return [[round(float(x), 1), round(float(y), 1), round(float(c), 2)] for x, y, c in k]


def run_poses(path: str, info, perc, progress=None) -> dict:
    """Поза для каждого человека «внутри кассы» в кадрах perception (те же моменты времени, ~3 к/с)."""
    from .video import iter_frames

    est = get_estimator()
    if not est.available():
        raise FileNotFoundError(f"pose model not found: {est.path}")
    times = list(perc.times)
    poses: list[list[dict]] = [[] for _ in times]
    if not times:
        return {"version": POSE_VERSION, "model": POSE_MODEL_NAME, "times": [], "poses": []}
    want = {}
    for j, t in enumerate(times):
        want[int(round(t * info.fps))] = j
    step = max(1, int(round(info.fps / float(getattr(perc, "sample_fps", 3.0) or 3.0))))
    total = max(1, info.frame_count)
    for idx, t, frame in iter_frames(path, every_n=step):
        j = want.get(idx)
        if j is None:
            j = int(np.argmin(np.abs(np.asarray(times) - t)))
            if abs(times[j] - t) > 0.5 / max(info.fps, 1.0) * step:
                continue
        obs = [o for o in perc.persons[j] if o.inside]
        if not obs:
            continue
        kps = est(frame, [o.xyxy for o in obs])
        poses[j] = [{"track_id": int(o.track_id), "kpts": _round_kpts(k)} for o, k in zip(obs, kps)]
        if progress and idx % (step * 30) == 0:
            progress("pose", int(100 * idx / total))
    return {"version": POSE_VERSION, "model": POSE_MODEL_NAME, "times": times, "poses": poses}


def load_or_compute(path: str, info, perc, cache_dir: Optional[Path] = None, progress=None) -> Optional[dict]:
    """Позы из кеша results/hands/<stem>.json (только если задан FEATURES_CACHE) или по видео (живой путь).
    Возвращает None, если модель позы недоступна (правила по рукам тогда молчат — «неизвестно», не «нарушение»)."""
    if not pose_enabled():
        return None
    stem = Path(path).stem.replace("X", "Х")
    if cache_dir is not None:
        hj = cache_dir / "hands" / f"{stem}.json"
        if hj.exists():
            try:
                d = json.loads(hj.read_text(encoding="utf-8"))
                ct = d.get("times") or []
                # кеш годится, только если посчитан для тех же кадров perception (треки/рамки те же)
                if d.get("version") == POSE_VERSION and len(ct) == len(perc.times) and np.allclose(ct, perc.times, atol=0.02):
                    return d
                log.warning("pose cache %s does not match perception frames — recomputing", hj)
            except Exception as exc:  # noqa: BLE001
                log.warning("pose cache %s unreadable: %s", hj, exc)
    if not get_estimator().available():
        log.warning("pose model %s not found — hand rules disabled", get_estimator().path)
        return None
    # кеш здесь только читается: пишет его scripts/collect_hands.py (загрузки сервиса не должны оседать в results/)
    return run_poses(path, info, perc, progress=progress)


# ---------------------------------------------------------------------------
# производные признаки (относительные, в системе тела)
# ---------------------------------------------------------------------------

def _vis(k: np.ndarray, i: int) -> bool:
    return bool(k[i, 2] >= KP_MIN_SCORE)


def body_frame(k: np.ndarray) -> Optional[tuple[np.ndarray, float, float]]:
    """(середина плеч, ширина плеч, длина торса) или None, если плечи не видны."""
    if not (_vis(k, L_SH) and _vis(k, R_SH)):
        return None
    sh_mid = (k[L_SH, :2] + k[R_SH, :2]) / 2.0
    sw = float(np.hypot(*(k[L_SH, :2] - k[R_SH, :2])))
    torso = None
    if _vis(k, L_HIP) and _vis(k, R_HIP):
        hip_mid = (k[L_HIP, :2] + k[R_HIP, :2]) / 2.0
        torso = float(np.hypot(*(hip_mid - sh_mid)))
    scale = max(sw, 0.6 * torso if torso else 0.0)
    if scale <= 1.0:
        return None
    return sh_mid, scale, (torso or 1.6 * scale)


def track_series(pose: dict, track_ids: list[int]) -> dict:
    """Ряды для набора треков (обычно — все треки кассира): по моментам pose['times'].

    wrists_rel[i] — (2, 2): координаты левой/правой кисти относительно середины плеч в единицах масштаба тела
                    (NaN, если кисть/плечи не видны); y растёт вниз (как в кадре).
    wrist_px[i]   — (2, 2) кисти в пикселях кадра (для зон, построенных от якорей), NaN если не видны.
    scale[i]      — масштаб тела (пиксели), NaN если нет.
    """
    times = np.asarray(pose.get("times") or [], float)
    n = len(times)
    rel = np.full((n, 2, 2), np.nan)
    px = np.full((n, 2, 2), np.nan)
    scale = np.full(n, np.nan)
    ids = set(int(t) for t in track_ids)
    for i, ps in enumerate(pose.get("poses") or []):
        best = None
        for p in ps:
            if int(p["track_id"]) in ids:
                k = np.asarray(p["kpts"], float)
                if best is None or k[:, 2].mean() > best[:, 2].mean():
                    best = k
        if best is None:
            continue
        bf = body_frame(best)
        for h, wi in enumerate((L_WR, R_WR)):
            if _vis(best, wi):
                px[i, h] = best[wi, :2]
                if bf is not None:
                    rel[i, h] = (best[wi, :2] - bf[0]) / bf[1]
        if bf is not None:
            scale[i] = bf[1]
    return {"t": times, "wrists_rel": rel, "wrist_px": px, "scale": scale}


def wrist_speed(series: dict) -> np.ndarray:
    """Скорость кистей (макс. из двух) в масштабах тела в секунду; NaN, где нельзя посчитать."""
    t = series["t"]
    rel = series["wrists_rel"]
    n = len(t)
    v = np.full(n, np.nan)
    for i in range(1, n):
        dt = t[i] - t[i - 1]
        if dt <= 0 or dt > 1.0:
            continue
        d = np.linalg.norm(rel[i] - rel[i - 1], axis=1)
        d = d[~np.isnan(d)]
        if len(d):
            v[i] = float(d.max()) / dt
    return v
