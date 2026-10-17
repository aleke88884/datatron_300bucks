"""Слой восприятия: детекции YOLOX на ~3 к/с, треки людей, признак «внутри кассы», статические якоря сцены.

Результат — Perception: по кадрам (t, люди, объекты) + треки + якоря (ноутбук, стулья, кандидаты на сейф, стол).
Роли (кассир/посетитель) и зоны строятся поверх в scene.py. Никаких лиц: только положение и движение.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .detect import Detection, get_detector
from .video import VideoInfo, iter_frames

log = logging.getLogger("pipeline.perception")

SAMPLE_FPS = 3.0
PERSON_CONF = 0.45
OBJECT_CONF = 0.3
TRACK_MAX_GAP_SEC = 1.2      # трек живёт без детекции до 1.2 с
TRACK_MATCH_IOU = 0.25
TRACK_MATCH_DIST = 0.6       # или центр ближе 0.6 высоты bbox
STATIC_CLASSES = {"laptop", "chair", "tv", "dining table", "refrigerator", "microwave", "oven"}
ITEM_CLASSES = {"handbag", "backpack", "suitcase", "cup", "bottle", "wine glass", "cell phone", "book", "remote", "bowl", "vase",
                "sandwich", "banana", "apple", "orange", "donut", "cake", "pizza", "hot dog"}   # еда — T8 (personal_items)


@dataclass
class PersonObs:
    t: float
    xyxy: tuple[float, float, float, float]
    score: float
    inside: bool
    track_id: int = -1
    state: int = 0        # +1 внутри / -1 снаружи / 0 спорно (до гистерезиса)

    @property
    def foot(self) -> tuple[float, float]:
        return ((self.xyxy[0] + self.xyxy[2]) / 2, self.xyxy[3])

    @property
    def h(self) -> float:
        return self.xyxy[3] - self.xyxy[1]


@dataclass
class Track:
    id: int
    obs: list[PersonObs] = field(default_factory=list)
    role: str = "unknown"   # cashier / visitor / outside

    @property
    def t0(self) -> float:
        return self.obs[0].t

    @property
    def t1(self) -> float:
        return self.obs[-1].t

    def at(self, t: float, tol: float = 0.6) -> Optional[PersonObs]:
        best = None
        for o in self.obs:
            if abs(o.t - t) <= tol and (best is None or abs(o.t - t) < abs(best.t - t)):
                best = o
        return best

    def inside_intervals(self, min_sec: float = 0.0) -> list[tuple[float, float]]:
        """Интервалы, когда человек внутри кассы."""
        out: list[list[float]] = []
        for o in self.obs:
            if not o.inside:
                continue
            if out and o.t - out[-1][1] <= TRACK_MAX_GAP_SEC + 0.01:
                out[-1][1] = o.t
            else:
                out.append([o.t, o.t])
        return [(a, b) for a, b in out if b - a >= min_sec]


@dataclass
class Perception:
    info: VideoInfo
    times: list[float]
    persons: list[list[PersonObs]]          # по кадрам
    objects: list[list[Detection]]          # по кадрам: статические + предметы
    tracks: list[Track]
    sample_fps: float = SAMPLE_FPS
    degrade: dict = field(default_factory=dict)   # {"stride": k, "stopped": bool, ...} — если сработал бюджет времени

    def frame_index(self, t: float) -> int:
        return int(np.argmin(np.abs(np.asarray(self.times) - t))) if self.times else 0

    def persons_inside_at(self, t: float) -> list[PersonObs]:
        return [p for p in self.persons[self.frame_index(t)] if p.inside]

    def count_inside(self) -> np.ndarray:
        return np.asarray([sum(1 for p in ps if p.inside) for ps in self.persons])


def _iou(a: tuple, b: tuple) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def inside_state(xyxy: tuple[float, float, float, float], info: VideoInfo, scene: "SceneGeometry") -> int:
    """+1 внутри, -1 снаружи, 0 неопределённо (дверной проём или ноги закрыты мебелью).

    Люди за стеклянной стеной видны только выше матовой полосы: их рамка обрезана по ней (низ < band_y).
    Люди внутри либо стоят на полу (низ ниже линии порога), либо закрыты столом/стулом — тогда низ рамки лежит
    на кромке мебели между полосой и порогом, но фигура крупнее, чем у человека за стеклом (перспектива).
    Правее дверной зоны стекла нет — всегда внутри.
    """
    x1, y1, x2, y2 = xyxy
    cx = (x1 + x2) / 2
    if cx / info.width >= scene.door_x_max:
        return 1
    band = scene.band_y * info.height
    thr = scene.threshold_y(cx, info)
    if y2 < band:
        return -1
    if y2 >= thr:
        return 1
    return 1 if (y2 - y1) >= scene.min_inside_h * info.height else 0


class SceneGeometry:
    """Геометрия камеры в нормированных координатах (доля ширины/высоты кадра).

    Порог двери — линия между двумя точками; всё, что ниже (foot_y больше), — внутри кассы. Значения по умолчанию
    откалиброваны по этой камере (дверь слева вверху, стол в середине); для другой камеры задаются через
    SCENE_JSON (путь к json с полями threshold_a, threshold_b, см. README).
    """

    def __init__(self, threshold_a=(0.0, 0.407), threshold_b=(0.55, 0.407), door_x_max=0.55, band_y=0.354, min_inside_h=0.165):
        self.threshold_a = threshold_a      # линия порога двери (ниже неё — пол кассы), нормированные координаты
        self.threshold_b = threshold_b
        self.door_x_max = door_x_max        # правее этой доли ширины стекла нет
        self.band_y = band_y                # низ матовой полосы на стекле: рамки людей снаружи обрезаны выше неё
        self.min_inside_h = min_inside_h    # минимальная высота фигуры (доля высоты кадра) для «внутри» в спорной зоне

    @classmethod
    def load(cls) -> "SceneGeometry":
        p = os.environ.get("SCENE_JSON")
        if p and Path(p).exists():
            d = json.loads(Path(p).read_text(encoding="utf-8"))
            return cls(
                tuple(d.get("threshold_a", (0.0, 0.407))), tuple(d.get("threshold_b", (0.55, 0.407))),
                float(d.get("door_x_max", 0.55)), float(d.get("band_y", 0.354)), float(d.get("min_inside_h", 0.165)),
            )
        return cls()

    def threshold_y(self, x: float, info: VideoInfo) -> float:
        (ax, ay), (bx, by) = self.threshold_a, self.threshold_b
        if x / info.width >= self.door_x_max:
            return by * info.height
        if bx == ax:
            return ay * info.height
        k = (by - ay) / (bx - ax)
        return (ay + k * (x / info.width - ax)) * info.height


def _match_tracks(active: list[Track], dets: list[PersonObs], t: float) -> None:
    """Жадное сопоставление детекций и треков по IoU / расстоянию центров."""
    pairs = []
    for ti, tr in enumerate(active):
        last = tr.obs[-1]
        if t - last.t > TRACK_MAX_GAP_SEC:
            continue
        for di, d in enumerate(dets):
            iou = _iou(last.xyxy, d.xyxy)
            lc = ((last.xyxy[0] + last.xyxy[2]) / 2, (last.xyxy[1] + last.xyxy[3]) / 2)
            dc = ((d.xyxy[0] + d.xyxy[2]) / 2, (d.xyxy[1] + d.xyxy[3]) / 2)
            dist = float(np.hypot(lc[0] - dc[0], lc[1] - dc[1])) / max(last.h, 1.0)
            if iou >= TRACK_MATCH_IOU or dist <= TRACK_MATCH_DIST:
                pairs.append((-(iou) + dist * 0.1, ti, di))
    pairs.sort()
    used_t, used_d = set(), set()
    for _, ti, di in pairs:
        if ti in used_t or di in used_d:
            continue
        used_t.add(ti)
        used_d.add(di)
        dets[di].track_id = active[ti].id
        active[ti].obs.append(dets[di])


def run_perception(path: str, info: VideoInfo, scene: Optional[SceneGeometry] = None, progress=None) -> Perception:
    from . import budget as _budget

    scene = scene or SceneGeometry.load()
    det = get_detector()
    bud = _budget.current()
    step = max(1, int(round(info.fps / SAMPLE_FPS)))
    times: list[float] = []
    persons: list[list[PersonObs]] = []
    objects: list[list[Detection]] = []
    tracks: list[Track] = []
    next_id = 1
    total = max(1, info.frame_count)
    gov = _budget.StrideGovernor(
        "восприятие", bud.perception_deadline() if bud else None, bud.perception_truncate_ts() if bud else None, budget=bud,
    )
    grid_total = max(1, total // step)
    for idx, t, frame in iter_frames(path, every_n=step, take=gov.take):
        dets = det(frame, conf=min(PERSON_CONF, OBJECT_CONF))
        ps: list[PersonObs] = []
        objs: list[Detection] = []
        for d in dets:
            if d.cls == "person" and d.score >= PERSON_CONF and d.h >= 0.06 * info.height:
                st = inside_state(d.xyxy, info, scene)
                ps.append(PersonObs(t=t, xyxy=d.xyxy, score=d.score, inside=st > 0, state=st))
            elif d.cls in STATIC_CLASSES or d.cls in ITEM_CLASSES:
                if d.score >= OBJECT_CONF:
                    objs.append(d)
        active = [tr for tr in tracks if t - tr.obs[-1].t <= TRACK_MAX_GAP_SEC]
        _match_tracks(active, ps, t)
        for p in ps:
            if p.track_id < 0:
                tr = Track(id=next_id, obs=[p])
                p.track_id = tr.id
                tracks.append(tr)
                next_id += 1
            elif p.state == 0:
                # спорная зона: наследуем предыдущее состояние трека (гистерезис)
                tr = next(tr for tr in tracks if tr.id == p.track_id)
                prev = tr.obs[-2] if len(tr.obs) >= 2 else None
                p.inside = bool(prev.inside) if prev is not None else False
        times.append(t)
        persons.append(ps)
        objects.append(objs)
        if progress and idx % (step * 15) == 0:
            progress("detection", int(100 * idx / total))
        if not gov.update(idx // step, grid_total):
            break
    perc = Perception(info=info, times=times, persons=persons, objects=objects, tracks=tracks)
    if gov.k > 1 or gov.stopped:
        perc.degrade = gov.summary()
        if bud is not None:
            bud.stats["perception"] = perc.degrade
    return perc


# ---------------------------------------------------------------- кэш (для отладки и тестов на датасете)

def perception_to_dict(p: Perception) -> dict:
    return {
        "times": p.times,
        "persons": [[{"t": o.t, "xyxy": o.xyxy, "score": o.score, "inside": o.inside, "track_id": o.track_id, "state": o.state} for o in ps] for ps in p.persons],
        "objects": [[{"cls": d.cls, "score": d.score, "xyxy": d.xyxy} for d in ds] for ds in p.objects],
    }


def perception_from_dict(d: dict, info: VideoInfo) -> Perception:
    persons = [[PersonObs(t=o["t"], xyxy=tuple(o["xyxy"]), score=o["score"], inside=o["inside"], track_id=o["track_id"], state=o.get("state", 0)) for o in ps] for ps in d["persons"]]
    objects = [[Detection(o["cls"], o["score"], tuple(o["xyxy"])) for o in ds] for ds in d["objects"]]
    tracks: dict[int, Track] = {}
    for ps in persons:
        for o in ps:
            tracks.setdefault(o.track_id, Track(id=o.track_id)).obs.append(o)
    return Perception(info=info, times=list(d["times"]), persons=persons, objects=objects, tracks=[tracks[k] for k in sorted(tracks)])


def cached_perception(path: str, info: VideoInfo, cache_dir: Optional[str] = None, progress=None) -> Perception:
    cache_dir = cache_dir or os.environ.get("PERCEPTION_CACHE")
    if cache_dir:
        key = Path(cache_dir) / (Path(path).stem + f"_{int(SAMPLE_FPS)}fps.json")
        if key.exists():
            try:
                return perception_from_dict(json.loads(key.read_text(encoding="utf-8")), info)
            except Exception as exc:  # noqa: BLE001
                log.warning("perception cache %s unreadable: %s", key, exc)
    p = run_perception(path, info, progress=progress)
    if cache_dir and not p.degrade:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        (Path(cache_dir) / (Path(path).stem + f"_{int(SAMPLE_FPS)}fps.json")).write_text(json.dumps(perception_to_dict(p)), encoding="utf-8")
    return p
