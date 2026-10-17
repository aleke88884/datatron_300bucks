"""Декодирование видео через OpenCV (бэкенд FFMPEG из imageio-ffmpeg, системный ffmpeg не нужен)."""
from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np

# Подкладываем ffmpeg из imageio-ffmpeg в PATH до импорта cv2 — на машинах со сломанным
# системным ffmpeg это гарантирует рабочий бэкенд. opencv-python-headless содержит собственный
# ffmpeg, так что это лишь страховка.
try:  # pragma: no cover
    import imageio_ffmpeg

    _ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    os.environ.setdefault("IMAGEIO_FFMPEG_EXE", _ffmpeg)
    os.environ["PATH"] = os.path.dirname(_ffmpeg) + os.pathsep + os.environ.get("PATH", "")
except Exception:  # noqa: BLE001
    _ffmpeg = None

import cv2  # noqa: E402


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_sec: float


class VideoDecodeError(RuntimeError):
    pass


_ascii_copies: dict[str, str] = {}


def _ascii_copy(path: str) -> Optional[str]:
    """Windows: OpenCV может не открыть путь с кириллицей — копия во временный файл с ASCII-именем."""
    if os.name != "nt" or path.isascii():
        return None
    if path not in _ascii_copies:
        import hashlib
        import shutil
        import tempfile

        dst = os.path.join(tempfile.gettempdir(), "cashdesk_" + hashlib.md5(path.encode("utf-8")).hexdigest()[:12] + os.path.splitext(path)[1])
        if not os.path.exists(dst):
            shutil.copy2(path, dst)
        _ascii_copies[path] = dst
    return _ascii_copies[path]


def release_temp_copies() -> None:
    """Удаляет временные ASCII-копии видео (Windows, кириллица в пути): тестовые видео не храним."""
    for src, dst in list(_ascii_copies.items()):
        try:
            os.remove(dst)
        except OSError:
            pass
        _ascii_copies.pop(src, None)


def open_capture(path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        alt = _ascii_copy(path)
        if alt:
            cap = cv2.VideoCapture(alt, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise VideoDecodeError(f"cannot open video: {path}")
    return cap


def probe(path: str) -> VideoInfo:
    cap = open_capture(path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()
    if fps <= 1 or fps > 240:
        fps = 30.0
    return VideoInfo(path=path, width=w, height=h, fps=fps, frame_count=n, duration_sec=(n / fps if n else 0.0))


def iter_frames(path: str, every_n: int = 1, take=None) -> Iterator[tuple[int, float, np.ndarray]]:
    """Итератор (индекс кадра, время в секундах, кадр BGR). Декодирует все кадры, отдаёт каждый every_n-й.
    take(n) — необязательный фильтр по номеру узла сетки n = idx // every_n (адаптивный шаг, см. budget.py):
    отброшенные узлы только «пролистываются» (grab без retrieve/конвертации)."""
    cap = open_capture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 1 or fps > 240:
        fps = 30.0
    idx = 0
    try:
        while True:
            if not cap.grab():
                break
            if idx % every_n == 0 and (take is None or take(idx // every_n)):
                ok, frame = cap.retrieve()
                if ok and frame is not None:
                    yield idx, idx / fps, frame
            idx += 1
    finally:
        cap.release()


def brightness_series(path: str, target_fps: float = 10.0) -> tuple[np.ndarray, np.ndarray, VideoInfo]:
    """Средняя яркость (0..255) по кадрам с частотой ~target_fps. Возвращает (times, values, info).
    Одновременно уточняет длительность по фактически декодированным кадрам."""
    info = probe(path)
    step = max(1, int(round(info.fps / target_fps)))
    times: list[float] = []
    vals: list[float] = []
    last_idx = -1
    for idx, t, frame in iter_frames(path, every_n=step):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        times.append(t)
        vals.append(float(gray.mean()))
        last_idx = idx
    if not times:
        raise VideoDecodeError(f"no frames decoded: {path}")
    # фактическое число кадров могло отличаться от заголовка
    decoded_frames = last_idx + step
    if info.frame_count <= 0 or abs(decoded_frames - info.frame_count) > step * 2:
        info.frame_count = decoded_frames
    info.duration_sec = round(info.frame_count / info.fps, 3)
    return np.asarray(times), np.asarray(vals), info


def read_frames_at(path: str, times: list[float], info: Optional[VideoInfo] = None) -> list[Optional[np.ndarray]]:
    """Кадры, ближайшие к моментам times (в том же порядке), за один последовательный проход.

    Позиционирование через CAP_PROP_POS_FRAMES на части роликов (переменный fps, Х-11, Х-23, Х-28, Х-37, Х-39,
    Х-45) попадает на несколько секунд позже нужного кадра. Детекции считаются последовательным декодированием
    (индекс кадра / fps), поэтому и кадры-доказательства читаем так же: grab() каждого кадра, retrieve() — только
    нужных. Стоимость — декодирование до последнего запрошенного момента."""
    if not times:
        return []
    if info is None:
        info = probe(path)
    last = max(info.frame_count - 1, 0)
    targets = [max(0, min(int(round(float(t) * info.fps)), last)) if last > 0 else max(0, int(round(float(t) * info.fps)))
               for t in times]
    want = set(targets)
    stop = max(targets)
    got: dict[int, np.ndarray] = {}
    prev: Optional[np.ndarray] = None
    prev_idx = -1
    cap = open_capture(path)
    try:
        idx = 0
        while idx <= stop:
            if not cap.grab():
                break
            if idx in want or idx == stop:
                ok, frame = cap.retrieve()
                if ok and frame is not None:
                    got[idx] = frame
                    prev, prev_idx = frame, idx
            idx += 1
        # запрошенный кадр за фактическим концом записи (заголовок завысил число кадров) — последний прочитанный
        if prev is None or idx <= stop:
            ok, frame = cap.retrieve()
            if ok and frame is not None:
                prev, prev_idx = frame, idx - 1
    finally:
        cap.release()
    out: list[Optional[np.ndarray]] = []
    for k in targets:
        if k in got:
            out.append(got[k])
        elif prev is not None and k >= prev_idx:
            out.append(prev)
        else:
            out.append(None)
    return out


def read_frame_at(path: str, t_sec: float, info: Optional[VideoInfo] = None) -> Optional[np.ndarray]:
    """Кадр, ближайший к t_sec (точный, последовательным декодированием; см. read_frames_at)."""
    return read_frames_at(path, [t_sec], info)[0]


def encode_jpeg_base64(frame: np.ndarray, quality: int = 82, max_side: Optional[int] = None) -> str:
    if max_side and max(frame.shape[:2]) > max_side:
        scale = max_side / max(frame.shape[:2])
        frame = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise VideoDecodeError("jpeg encode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")
