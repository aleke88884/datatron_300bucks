"""Детектор объектов YOLOX (Apache-2.0) через ONNX Runtime, CPU.

Модель: официальный экспорт YOLOX-m (COCO, 80 классов) из релиза Megvii-BaseDetection/YOLOX 0.1.1rc0.
Вес не хранится в git: scripts/get_models.sh скачивает его в service/models/ (в Docker — на этапе сборки).
Вход 640×640 letterbox (заливка 114), BGR без нормализации — как в demo/ONNXRuntime официального репозитория.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

MODELS_DIR = Path(os.environ.get("MODELS_DIR", Path(__file__).resolve().parent.parent / "models"))
YOLOX_FILE = os.environ.get("YOLOX_MODEL", "yolox_m.onnx")
INPUT_SIZE = int(os.environ.get("YOLOX_INPUT", "640"))

COCO_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)


@dataclass
class Detection:
    cls: str
    score: float
    xyxy: tuple[float, float, float, float]

    @property
    def cx(self) -> float:
        return (self.xyxy[0] + self.xyxy[2]) / 2

    @property
    def cy(self) -> float:
        return (self.xyxy[1] + self.xyxy[3]) / 2

    @property
    def w(self) -> float:
        return self.xyxy[2] - self.xyxy[0]

    @property
    def h(self) -> float:
        return self.xyxy[3] - self.xyxy[1]

    @property
    def foot(self) -> tuple[float, float]:
        """Точка опоры: середина нижней грани bbox (для людей — где стоят ноги)."""
        return ((self.xyxy[0] + self.xyxy[2]) / 2, self.xyxy[3])

    @property
    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)


def _letterbox(img: np.ndarray, size: int) -> tuple[np.ndarray, float]:
    padded = np.full((size, size, 3), 114, dtype=np.uint8)
    r = min(size / img.shape[0], size / img.shape[1])
    rw, rh = int(img.shape[1] * r), int(img.shape[0] * r)
    resized = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_LINEAR)
    padded[:rh, :rw] = resized
    return padded, r


def _decode(outputs: np.ndarray, size: int) -> np.ndarray:
    """Сетки/страйды YOLOX: [N, 85] → (cx, cy, w, h) в пикселях входа + objectness + 80 классов."""
    grids, strides = [], []
    for stride in (8, 16, 32):
        hs = ws = size // stride
        yv, xv = np.meshgrid(np.arange(hs), np.arange(ws), indexing="ij")
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        strides.append(np.full((1, grid.shape[1], 1), stride))
    grids = np.concatenate(grids, 1)
    strides = np.concatenate(strides, 1)
    outputs = outputs.copy()
    outputs[..., :2] = (outputs[..., :2] + grids) * strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * strides
    return outputs[0]


def _nms(boxes: np.ndarray, scores: np.ndarray, thr: float) -> list[int]:
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(ovr <= thr)[0] + 1]
    return keep


class YoloxDetector:
    """Потокобезопасная обёртка ONNX Runtime. Создаётся один раз на процесс (pipeline.warmup)."""

    def __init__(self, model_path: Optional[Path] = None, input_size: int = INPUT_SIZE, threads: Optional[int] = None):
        import onnxruntime as ort

        self.path = Path(model_path or MODELS_DIR / YOLOX_FILE)
        if not self.path.exists():
            raise FileNotFoundError(f"нет весов детектора {self.path}; запустите scripts/get_models.sh")
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(self.path), so, providers=[p.strip() for p in os.environ.get("ORT_PROVIDERS", "CPUExecutionProvider").split(",") if p.strip()])
        self.input_name = self.session.get_inputs()[0].name
        self.size = input_size
        self._lock = threading.Lock()

    def __call__(self, frame: np.ndarray, conf: float = 0.3, nms: float = 0.45, classes: Optional[set[str]] = None) -> list[Detection]:
        img, r = _letterbox(frame, self.size)
        blob = img.astype(np.float32).transpose(2, 0, 1)[None]
        with self._lock:
            out = self.session.run(None, {self.input_name: blob})[0]
        pred = _decode(out, self.size)
        boxes = pred[:, :4]
        scores = pred[:, 4:5] * pred[:, 5:]
        cls_idx = scores.argmax(1)
        cls_score = scores[np.arange(len(scores)), cls_idx]
        keep = cls_score > conf
        boxes, cls_idx, cls_score = boxes[keep], cls_idx[keep], cls_score[keep]
        if len(boxes) == 0:
            return []
        xyxy = np.empty_like(boxes)
        xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        xyxy /= r
        H, W = frame.shape[:2]
        xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, W - 1)
        xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, H - 1)
        dets: list[Detection] = []
        # NMS по классам
        for c in np.unique(cls_idx):
            name = COCO_CLASSES[int(c)]
            if classes is not None and name not in classes:
                continue
            m = cls_idx == c
            for k in _nms(xyxy[m], cls_score[m], nms):
                b = xyxy[m][k]
                dets.append(Detection(name, float(cls_score[m][k]), (float(b[0]), float(b[1]), float(b[2]), float(b[3]))))
        dets.sort(key=lambda d: -d.score)
        return dets


_detector: Optional[YoloxDetector] = None
_detector_lock = threading.Lock()


def get_detector() -> YoloxDetector:
    global _detector
    with _detector_lock:
        if _detector is None:
            _detector = YoloxDetector()
        return _detector
