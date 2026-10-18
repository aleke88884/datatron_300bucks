"""T13 — дверь кассового помещения оставлена открытой (по признакам pipeline/door.py).

Физика правила (одна фраза): человек прошёл через дверь кассы и дверь после этого долго не возвращается в закрытое
состояние, хотя у двери уже никого нет, — касса не изолирована от общего помещения.

Чем отличается от правильного поведения (по датасету): в чистых роликах дверь закрывают за 2–4 с после прохода
(Х-31, Х-3, Х-16, Х-35, Х-37, выходы в Х-25/Х-27/Х-29); в нарушениях (Х-32, Х-33, Х-30) она остаётся открытой
10–17 с до конца записи (открытый участок в чистых — не больше 2.8 с). Порог OPEN_HOLD_SEC = 6 с — с запасом в обе стороны.

Защита от ложных:
  * «открыта» — изменение не меньше OPEN_RATIO от пика первого открывания (неплотно прикрытая дверь в чистом Х-39 даёт
    ~0.5 пика — не открыта) и не меньше OPEN_MIN;
  * кадры с выключенным светом / резкой сменой экспозиции (яркость кадра вне BRIGHT_OK от эталона) не учитываются;
  * полярность: если человек вошёл через проём раньше первого движения створки при «эталонном» виде области, то
    эталон и был открытой дверью (запись началась с открытой двери, движение створки — закрывание) — правило молчит;
  * в открытом участке ≥ MIN_VALID_FRAC кадров видны (не закрыты людьми).

Таймкод — момент прохода через дверной проём (наибольшее перекрытие области двери рамкой вошедшего), т. е. начало
действия «вошёл и не закрыл дверь».
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .context import Event, VideoContext

OPEN_RATIO = 0.75          # открыта: изменение ≥ 75 % пика первого открывания…
OPEN_MIN = 0.35            # …и ≥ 35 % пикселей области двери
OPEN_HOLD_SEC = 6.0        # дверь непрерывно открыта не менее 6 с (чистые: ≤ 2.8 с, нарушения: ≥ 10 с)
CLOSE_CONFIRM = 2          # «закрыта» подтверждается двумя подряд валидными кадрами ниже порога (0.5 с при 4 к/с)
MIN_VALID_FRAC = 0.6       # в открытом участке ≥ 60 % кадров видны (не закрыты людьми) и валидны
BRIGHT_OK = (0.8, 1.25)    # яркость кадра относительно эталона: вне диапазона — свет/экспозиция, кадр не судим
CROSS_MIN_COVER = 0.05     # проход через проём: рамка человека перекрывает ≥ 5 % области двери
CROSS_SEARCH = (-1.0, 4.0) # проход, открывший дверь, ищется в окне от начала открытого участка
POLARITY_FRAC = 0.5        # при проходе изменение < 0.5 порога → эталон был «открытой дверью»


def _series(d: dict, ctx: VideoContext) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.asarray(d["t"], float)
    ch = np.asarray([np.nan if v is None else float(v) for v in d["change"]], float)
    br = np.asarray(d.get("bright_ratio") or [1.0] * len(t), float)
    valid = ~np.isnan(ch) & (br >= BRIGHT_OK[0]) & (br <= BRIGHT_OK[1])
    sc = getattr(ctx, "scene", None)
    if sc is not None and len(getattr(sc, "t", [])) and len(sc.lights_off) == len(sc.t):
        lo = np.asarray([bool(sc.lights_off[int(np.argmin(np.abs(sc.t - ti)))]) for ti in t])
        valid &= ~lo
    return t, ch, valid


def _nearest_valid(t: np.ndarray, ch: np.ndarray, valid: np.ndarray, ti: float, tol: float = 0.75) -> Optional[float]:
    idx = np.flatnonzero(valid & (np.abs(t - ti) <= tol))
    if not len(idx):
        return None
    return float(ch[idx[np.argmin(np.abs(t[idx] - ti))]])


def open_runs(d: dict, ctx: VideoContext) -> tuple[list[tuple[int, int, float]], float, tuple]:
    """Открытые участки (i0, i1, доля валидных) и порог «открыта»."""
    t, ch, valid = _series(d, ctx)
    peak = d.get("peak") or 0.0
    thr = max(OPEN_MIN, OPEN_RATIO * float(peak))
    is_open = valid & (ch >= thr)
    is_closed = valid & (ch < thr)
    runs = []
    i, n = 0, len(t)
    while i < n:
        if not is_open[i]:
            i += 1
            continue
        j, last_open, closed_streak = i, i, 0
        while j < n:
            if is_closed[j]:
                closed_streak += 1
                if closed_streak >= CLOSE_CONFIRM:
                    break
            elif is_open[j]:
                closed_streak = 0
                last_open = j
            j += 1
        seg = slice(i, last_open + 1)
        vf = float(valid[seg].mean()) if last_open >= i else 0.0
        runs.append((i, last_open, vf))
        i = j + 1 if j < n else n
    return runs, thr, (t, ch, valid)


def polarity_suspect(d: dict, crossings: list[dict], t, ch, valid, thr: float) -> bool:
    """Эталон снят с открытой двери? Пройти в кассу через закрытую дверь нельзя: если кто-то вошёл через проём раньше
    первого движения створки, а область при этом выглядела как эталон, то эталон — открытая дверь (запись началась
    с открытой двери, а движение створки — это её закрывание)."""
    tm0 = float(d.get("t_first_motion") or 0.0)
    for c in crossings:
        if c["kind"] != "in" or float(c["t_cross"]) >= tm0 - 0.5:
            continue
        v = _nearest_valid(t, ch, valid, float(c["t_cross"]))
        if v is None or v < POLARITY_FRAC * thr:
            return True
    return False


def rule_t13_door(ctx: VideoContext) -> list[Event]:
    d = (getattr(ctx, "cache", None) or {}).get("door")
    if not d or not d.get("found") or not d.get("peak"):
        return []
    runs, thr, (t, ch, valid) = open_runs(d, ctx)
    if not runs:
        return []
    crossings = [c for c in d.get("crossings", []) if c.get("t_cross") is not None and c.get("cover", 0) >= CROSS_MIN_COVER]
    if polarity_suspect(d, crossings, t, ch, valid, thr):
        ctx.cache.setdefault("door_notes", []).append(
            "вход через проём до первого движения створки — эталон мог быть открытой дверью, T13 не применяется")
        return []
    sc = getattr(ctx, "scene", None)
    for i0, i1, vf in runs:
        t0, t1 = float(t[i0]), float(t[i1])
        dur = t1 - t0
        if dur < OPEN_HOLD_SEC or vf < MIN_VALID_FRAC:
            continue
        opener = [c for c in crossings if t0 + CROSS_SEARCH[0] <= float(c["t_cross"]) <= t0 + CROSS_SEARCH[1]]
        opener = max(opener, key=lambda c: c["cover"]) if opener else None
        t_ev = float(opener["t_cross"]) if opener else t0
        later = [c for c in crossings if c["kind"] == "in" and (opener is None or c["track"] != opener["track"])
                 and t_ev + 1.0 < float(c["t_cross"]) <= t1 + 0.5]
        till_end = i1 >= len(t) - 3
        empty = False
        if sc is not None and len(getattr(sc, "t", [])):
            m = (sc.t >= t0) & (sc.t <= t1)
            empty = bool(m.any() and (sc.n_inside[m] == 0).mean() >= 0.8)
        till = f"до конца записи ({t1:.1f} с)" if till_end else f"до {t1:.1f} с"
        roles = getattr(sc, "roles", None) or {}
        who = {"cashier": "кассир", "visitor": "посетитель"}.get(roles.get(opener["track"]) if opener else "", "человек")
        if empty:
            what = f"В {t_ev:.1f} с дверь кассового помещения открыта, внутри никого нет; дверь остаётся открытой {till}, не менее {dur:.1f} с."
        elif opener and opener["kind"] == "out":
            what = (f"В {t_ev:.1f} с {who} выходит из кассового помещения через дверь и не закрывает её: дверь остаётся "
                    f"открытой {till}, не менее {dur:.1f} с.")
        else:
            what = (f"В {t_ev:.1f} с {who} входит в кассовое помещение и не закрывает за собой дверь: дверь остаётся "
                    f"открытой {till}, не менее {dur:.1f} с.")
        if later:
            what += f" В {float(later[0]['t_cross']):.1f} с через открытую дверь в помещение проходит ещё один человек."
        desc = what + (" Может указывать на то, что вход в кассовый узел не закрыт (п. 8 Правил № 14: кассовый узел "
                       "изолируется закрывающимся служебным входом). Закрывалась ли дверь на замок, по видео не "
                       "устанавливается; признак для проверки человеком.")
        conf = min(0.8, 0.66 + 0.015 * min(10.0, dur - OPEN_HOLD_SEC) + (0.05 if later else 0.0))
        x1, y1, x2, y2 = (int(round(v)) for v in d["roi"])
        dur_v = max(0.1, ctx.info.duration_sec - 0.1)
        times = [min(t_ev, dur_v), min(t_ev + 3.0, t1, dur_v), min(max(t_ev + 6.0, t1 - 0.5), dur_v)]
        return [Event(
            type_code="T13", timecode_sec=round(t_ev, 2), end_sec=round(t1, 2), confidence=round(conf, 2),
            description=desc, evidence_times=times,
            evidence_captions=["проход через дверь кассы", "дверь кассы остаётся открытой", "дверь по-прежнему открыта"],
            extras={"evidence_boxes": [(x1, y1, x2, y2)] * 3, "open_sec": dur, "door_thr": thr, "empty": empty,
                    "opener": opener, "later": later[:1]},
        )]
    return []
