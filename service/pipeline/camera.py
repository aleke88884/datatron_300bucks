"""Самокалибровка камеры по самому видео: масштаб сцены, экспозиция и тест «человек внутри кассы».

Финальное тестовое видео снято с другого ракурса, поэтому всё, что завязано на пиксели этой камеры, нужно
выводить из данных ролика. Модуль ничего не знает о конкретных координатах помещения; он опирается на:

  * масштаб людей — высота рамки уверенно детектированного человека (75-й перцентиль по ролику);
  * столешницу (pipeline.states.desk_region, по цвету дерева) — её центр задаёт уровень «пола кассы»;
  * среднюю яркость кадра — для приведения цветовых и яркостных порогов к экспозиции эталонной камеры.

Тест «внутри» без геометрии двери: люди за стеклянной стеной (снаружи кассы) находятся за столом кассира,
дальше от камеры, и нижняя кромка их рамки лежит заметно выше уровня столешницы; у людей внутри (стоят на
полу кассы или сидят за столом) низ рамки — на уровне столешницы или ниже. Смещение низа рамки относительно
центра стола, делённое на рост человека в кадре, по 42 обучающим видео: у людей снаружи ≤ −0.59 (99 %),
у людей внутри ≥ −0.46 (99 %). Пороги: < −0.70 — снаружи, > −0.45 — внутри; в спорной полосе крупная фигура
(≥ 0.8 роста — ближе к камере, чем стекло) внутри, иначе состояние наследуется по треку (через стекло не
проходят). Признак не зависит от разрешения, отражения, сдвига и умеренного поворота кадра; с откалиброванной
геометрией этой камеры он совпадает на 99.9 % уверенных наблюдений (расходятся 11 из 8006).

Режимы (CAMERA_INSIDE): ``auto`` (по умолчанию) — откалиброванная геометрия эталонной камеры
(perception.SceneGeometry) остаётся, если она согласуется с относительным тестом на этом ролике, иначе
используется относительный тест; ``fixed`` — всегда геометрия; ``relative`` — всегда относительный тест.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

from .video import VideoInfo

log = logging.getLogger("pipeline.camera")

# --- эталон: камера обучающего датасета (464×848). Это не координаты, а единицы измерения: пороги в пикселях
#     по всему пайплайну заданы для этого кадра и пересчитываются через scale().
REF_FRAME_H = 848.0
REF_PERSON_H = 0.225 * REF_FRAME_H   # медианный «рост» человека в кадре (75-й перцентиль высот рамок по ролику)
REF_BRIGHTNESS = 137.5               # медианная средняя яркость кадра по 42 роликам (133–144)

STRONG_SCORE = 0.62                  # уверенная детекция человека (как в scene.py)
SCALE_RANGE = (0.5, 1.4)             # масштаб по росту людей заменяет H/848, только если отношение вне диапазона:
                                     # снизу 0.5 — в роликах, где виден только сидящий кассир, «рост» ≈ 0.6 эталона
SCALE_MIN_OBS = 8                    # минимум уверенных наблюдений людей для оценки масштаба
EXPOSURE_DEADZONE = (0.85, 1.18)     # в пределах разброса обучающих роликов коррекция экспозиции не применяется
EXPOSURE_CLIP = (0.5, 2.2)

INSIDE_OUT_REL = -0.70               # низ рамки выше центра стола более чем на 0.70 роста — снаружи (за стеклом)
INSIDE_IN_REL = -0.45                # ниже −0.45 роста — внутри; между порогами — спорно (наследуется по треку)
INSIDE_BIG_H = 0.80                  # в спорной полосе фигура ростом ≥ 0.8 эталонного — внутри: человек ближе к камере,
                                     # чем стеклянная стена (люди за стеклом в кадре ~0.4–0.5 роста, 95 % ≤ 0.75)
AUTO_MAX_CONFLICT = 0.05             # доля уверенных наблюдений, где геометрия и относительный тест (с гистерезисом)
                                     # расходятся: на 42 обучающих роликах ≤ 0.042 (≤ 2 наблюдений), зеркальный кадр 0.07–0.19
AUTO_MIN_CONFLICT = 3                # …и не меньше 3 наблюдений (1 с при 3 к/с), чтобы сменить режим


def inside_mode() -> str:
    m = os.environ.get("CAMERA_INSIDE", "auto").strip().lower()
    return m if m in ("auto", "fixed", "relative") else "auto"


# ---------------------------------------------------------------- масштаб

def person_height(perc) -> Optional[float]:
    """Рост человека в кадре, пикс.: 75-й перцентиль высот уверенных рамок (стоящие люди, не сидящий кассир)."""
    hs = [o.h for ps in perc.persons for o in ps if o.score >= STRONG_SCORE]
    if len(hs) < SCALE_MIN_OBS:
        return None
    return float(np.percentile(hs, 75))


def scene_scale(perc, info: VideoInfo) -> float:
    """Множитель для порогов, заданных в пикселях эталонного кадра 464×848.

    По умолчанию H/848 (точно для эталонной камеры и её перекодировок в другом разрешении). Если рост людей
    в кадре расходится с этим значением сильнее, чем допускает SCALE_RANGE (другая камера, приближение,
    альбомный кадр), масштаб берётся по росту людей.
    """
    s_h = info.height / REF_FRAME_H if info.height else 1.0
    ph = person_height(perc) if perc is not None else None
    if ph is None or s_h <= 0:
        return s_h
    s_p = ph / REF_PERSON_H
    r = s_p / s_h
    if SCALE_RANGE[0] <= r <= SCALE_RANGE[1]:
        return s_h
    return float(np.clip(s_p, 0.25, 4.0))


# ---------------------------------------------------------------- экспозиция

def exposure_level(gray: np.ndarray) -> float:
    """Уровень экспозиции кадра: средняя яркость без почти чёрных пикселей (поля/углы после поворота кадра,
    чёрные рамки альбомного видео не должны выглядеть как недоэкспонированная сцена)."""
    v = gray[gray > 10]
    return float(v.mean()) if v.size >= 0.2 * gray.size else float(gray.mean())


def exposure_gain(median_brightness: Optional[float]) -> float:
    """Во сколько раз привести яркость к эталонной экспозиции (для цветовых/яркостных порогов).

    1.0 внутри EXPOSURE_DEADZONE (обучающие ролики не меняются); иначе REF/median, ограниченный EXPOSURE_CLIP.
    """
    if not median_brightness or median_brightness <= 1:
        return 1.0
    g = REF_BRIGHTNESS / float(median_brightness)
    if EXPOSURE_DEADZONE[0] <= g <= EXPOSURE_DEADZONE[1]:
        return 1.0
    return float(np.clip(g, *EXPOSURE_CLIP))


def apply_gain_hsv(hsv: np.ndarray, gain: float) -> np.ndarray:
    """Канал V (яркость) HSV-кадра, приведённый к эталонной экспозиции; оттенок и насыщенность не меняются."""
    if gain == 1.0:
        return hsv
    out = hsv.copy()
    out[..., 2] = np.clip(out[..., 2].astype(np.float32) * gain, 0, 255).astype(np.uint8)
    return out


# ---------------------------------------------------------------- тест «внутри»

def relative_state(xyxy, desk_cy: float, href: float) -> int:
    """+1 внутри / −1 снаружи / 0 спорно — по смещению низа рамки от центра стола в единицах роста."""
    rel = (float(xyxy[3]) - desk_cy) / max(href, 1.0)
    if rel < INSIDE_OUT_REL:
        return -1
    if rel > INSIDE_IN_REL:
        return 1
    if (float(xyxy[3]) - float(xyxy[1])) >= INSIDE_BIG_H * href:
        return 1
    return 0


def _labels(perc, states_fn) -> list[tuple[object, int, bool]]:
    """(наблюдение, состояние, внутри) по тесту states_fn с гистерезисом по треку (как в run_perception):
    спорное состояние наследует предыдущее значение трека, у нового трека — «снаружи»."""
    last: dict[int, bool] = {}
    out = []
    for ps in perc.persons:
        for o in ps:
            st = states_fn(o)
            ins = True if st > 0 else (False if st < 0 else last.get(o.track_id, False))
            last[o.track_id] = ins
            out.append((o, st, ins))
    return out


def _relabel(perc, states_fn) -> None:
    """Переписывает PersonObs.inside/state по новому тесту (с гистерезисом по треку)."""
    for o, st, ins in _labels(perc, states_fn):
        o.state, o.inside = st, ins


def calibrate(perc, states: dict, info: VideoInfo) -> dict:
    """Калибровка по ролику (вызывается analyze_video до построения сцены; работает и на кеше признаков).

    Возвращает сводку {mode, scale, gain, conflict, ...} и при необходимости переписывает perc «внутри/снаружи».
    """
    sc = (states or {}).get("scene") or {}
    desk = sc.get("desk")
    scale = sc.get("scale") or scene_scale(perc, info)
    gain = sc.get("exposure_gain")
    if gain is None:   # кеш признаков старого формата: уровень экспозиции по средней яркости кадра
        gain = exposure_gain(sc.get("exposure_level") or sc.get("median_brightness"))
    href = person_height(perc)
    out = {"mode": "fixed", "scale": round(float(scale), 3), "gain": round(float(gain), 3),
           "person_h": None if href is None else round(href / max(info.height, 1), 3), "conflict": None}
    want = inside_mode()
    if want == "fixed":
        return out
    if desk is None or href is None:
        if want == "relative":
            # без стола относительный тест невозможен: внутри — все уверенные детекции (роли/фантомы — в scene.py)
            _relabel(perc, lambda o: 1)
            out["mode"] = "all-inside"
        out["residual"] = "no-desk" if desk is None else "no-people"
        return out
    desk_cy = float(desk["center"][1])
    rel = _labels(perc, lambda o: relative_state(o.xyxy, desk_cy, href))
    strong = [(o, ins) for o, _st, ins in rel if o.score >= STRONG_SCORE]
    conflict = sum(1 for o, ins in strong if o.inside != ins)
    frac = conflict / max(1, len(strong))
    out["conflict"] = round(frac, 3)
    out["residual"] = round(frac, 3)
    if want == "relative" or (conflict >= AUTO_MIN_CONFLICT and frac > AUTO_MAX_CONFLICT):
        _relabel(perc, lambda o: relative_state(o.xyxy, desk_cy, href))
        out["mode"] = "relative"
    return out


__all__ = [
    "REF_FRAME_H", "REF_PERSON_H", "REF_BRIGHTNESS", "calibrate", "scene_scale", "exposure_gain",
    "apply_gain_hsv", "relative_state", "person_height", "inside_mode", "exposure_level",
]
