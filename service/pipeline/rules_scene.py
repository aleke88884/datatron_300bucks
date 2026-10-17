"""Правила T1, T2, T4–T8, T10, T13 на таблице состояний сцены (pipeline.scene.Scene).

Каждое правило: rule_tN(ctx) -> list[Event]. ctx.scene заполняет analyze_video (или scripts/eval_rules.py
из сохранённых признаков). Таймкод события = момент перехода состояния (ПЛАН 5.4), описание — только
наблюдаемое (ПЛАН 4.4). Пороги подобраны по 42 обучающим видео (scripts/eval_rules.py) с условием
«ноль ложных срабатываний на чистых».
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .context import Event, VideoContext
from .scene import DESK_OBJ_MIN, GRID_FPS, Scene

DT = 1.0 / GRID_FPS

# --- пороги -------------------------------------------------------------
T1_AWAY_SEC = 2.0            # кассир вне зоны стола не менее 2 с
T1_PRESENT_BEFORE_SEC = 1.0  # ценности лежали на столе уже за 1 с до ухода
T2_OPEN_SEC = 1.5            # дверца сейфа видна открытой (без перекрытия) не менее 1.5 с
T2_SCORE = 1.0               # safe_open_score ≥ 1.0 (прирост тёмного ≥ SAFE_OPEN_DELTA)
T2_ENABLED = False           # признак «тёмный проём у сейфа» растёт и в чистых роликах, когда кассир стоит у сейфа (X-21, X-23, X-25, X-27) — выключено до классификатора дверцы
T3_AWAY_SEC = 2.0            # экран активен, кассир вне зоны стола не менее 2 с
T4_SEC = 1.5                 # ≥ 3 уверенных человек внутри не менее 1.5 с
T5_SEC = 1.0                 # посетитель в рабочей зоне не менее 1 с
T6_AFTER_SEC = 4.0           # окно после ухода посетителя
T6_MIN_FRAC = 0.5            # доля окна, когда на столе остаются наличные/ценности
T6_ENABLED = False           # признак «предмет на столе» не отличает деньги от документов клиента (ложные X-5, X-19) — выключено
T7_BEFORE_SEC = 3.0          # наличные лежали открыто не менее 3 с до входа
T7_MIN_SAT = 70              # купюры — насыщенный цвет (синие/зелёные/фиолетовые), документы и тени — нет
T7_MIN_AREA = 200
T7_MIN_VAL = 100             # купюры светлые; тёмные предметы (кабель, тень ноутбука) — нет
T7_CASH_GAP_SEC = 2.5        # пачку купюр переложили / на мгновение закрыли рукой — она всё ещё на столе
T8_DRINK_MIN_DET = 3
T8_DRINK_MIN_SCORE = 0.5
T8_BAG_MIN_DET = 4
T8_BAG_MIN_SCORE = 0.3
T10_VISITOR_SEC = 1.0
T10_WINDOW_SEC = 5.0         # сумка на столе не позже чем через 5 с после входа посетителя
T13_EMPTY_SEC = 1.5
T13_ENABLED = False          # чистое Х-31 тоже начинается с пустой кассы — признак не разделяет, выключено


# --- утилиты --------------------------------------------------------------

def _scene(ctx: VideoContext) -> Optional[Scene]:
    return getattr(ctx, "scene", None)


def _obj_present(sc: Scene) -> np.ndarray:
    return (sc.desk_obj_area >= DESK_OBJ_MIN) | sc.bag_on_desk


def _cash(sc: Scene, fill_gap_sec: float = 0.0) -> np.ndarray:
    """Пачка купюр, сливающаяся с деревом, лежит на столе (pipeline/cashdetect.py); разрывы ≤ fill_gap_sec
    (кассир переложил пачку, рука на мгновение закрыла её) заполняются."""
    m = sc.cash_present if len(sc.cash_present) == len(sc.t) else np.zeros(len(sc.t), bool)
    if fill_gap_sec > 0 and m.any():
        out = np.zeros(len(sc.t), bool)
        for a, b in sc.runs(m, 0.0, fill_gap_sec=fill_gap_sec):
            out[a:b + 1] = True
        m = out
    return m


def _cash_box(sc: Scene, i: int) -> Optional[tuple[int, int, int, int]]:
    """Рамка пачки купюр около индекса i — с полями, чтобы маленький предмет был заметен на кадре-доказательстве."""
    if len(sc.cash_boxes) == len(sc.t):
        for k in (i, i - 1, i + 1, i - 2, i + 2):
            if 0 <= k < len(sc.t) and sc.cash_boxes[k]:
                x1, y1, x2, y2 = sc.cash_boxes[k]
                p = max(6.0 * sc.info.height / 848.0, 0.4 * max(x2 - x1, y2 - y1))
                return (int(max(0, x1 - p)), int(max(0, y1 - p)),
                        int(min(sc.info.width - 1, x2 + p)), int(min(sc.info.height - 1, y2 + p)))
    return None


def _obj_box(sc: Scene, i: int) -> Optional[tuple[int, int, int, int]]:
    if sc.desk_objs[i]:
        b = max(sc.desk_objs[i], key=lambda o: o["area"])["bbox"]
        return tuple(int(v) for v in b)
    return None


def _desk_zone(sc: Scene) -> Optional[tuple[float, float, float, float]]:
    d = sc.anchors.get("desk")
    if not d:
        return None
    x1, y1, x2, y2 = d["bbox"]
    s = sc.info.height / 848.0
    return (x1 - 10 * s, y1 - 110 * s, x2 + 10 * s, y2 + 10 * s)


def _overlap(a, b) -> bool:
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def _iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _leave_runs(sc: Scene, gone_sec: float = 1.0, away_sec: float = 5.0) -> list[tuple[int, int, int]]:
    """Уходы кассира: (i_leave, a, b) — начало ухода от стола, начало и конец участка «ушёл».
    «Ушёл» = вне помещения ≥ gone_sec (участок у конца записи — ≥ 0.5 с) или вне зоны стола ≥ away_sec."""
    n = len(sc.t)
    out = []
    gone = ~sc.cashier_inside
    away = ~sc.cashier_at_desk
    runs = []
    for a, b in sc.runs(gone, 0.0, fill_gap_sec=1.0):
        dur = (b - a + 1) * DT
        if dur >= gone_sec or (b >= n - 1 and dur >= 0.5):
            runs.append((a, b))
    for a, b in sc.runs(away, away_sec, fill_gap_sec=1.0):
        if not any(ra <= a <= rb or a <= ra <= b for ra, rb in runs):
            runs.append((a, b))
    runs.sort()
    for a, b in runs:
        # был ли кассир вообще до ухода
        if not sc.cashier_inside[:a].any():
            continue
        i_leave = a
        while i_leave > 0 and not sc.cashier_at_desk[i_leave - 1] and a - i_leave < int(8 * GRID_FPS):
            i_leave -= 1
        k = sc.stood_up_before(i_leave)
        if k is not None and k < i_leave:
            i_leave = k
        out.append((i_leave, a, b))
    return out


def _clip(sc: Scene, t: float) -> float:
    return float(max(0.0, min(t, sc.info.duration_sec - 0.1)))


def _ev_times(sc: Scene, t0: float, offsets=(0.0, 1.5, 3.5)) -> list[float]:
    return [_clip(sc, t0 + o) for o in offsets]


# --- T1 -------------------------------------------------------------------

def rule_t1(ctx: VideoContext) -> list[Event]:
    """Наличные/ценности оставлены без присмотра: кассир ушёл, на столе остаются деньги/сумка."""
    sc = _scene(ctx)
    if sc is None or len(sc.t) == 0:
        return []
    cash = _cash(sc)
    present = _obj_present(sc) | cash
    any_obj = sc.desk_obj_area_any if len(sc.desk_obj_area_any) == len(sc.t) else sc.desk_obj_area
    present_any = (any_obj >= DESK_OBJ_MIN) | sc.bag_on_desk | cash
    lit = ~sc.lights_off
    events: list[Event] = []
    for i_leave, a, b in _leave_runs(sc):
        if sc.lights_off[max(0, i_leave - int(2 * GRID_FPS)):b + 1].any():
            continue   # в темноте детекции людей и предметов ненадёжны; уход в темноте покрывает T3 (экран светится)
        # состояние стола берём по последним освещённым кадрам перед уходом (предмет мог быть перекрыт рамкой
        # стоящего рядом кассира — учитываем и такие) и в начале ухода (без перекрытия)
        lo = max(0, i_leave - int(T1_PRESENT_BEFORE_SEC * GRID_FPS))
        before = present_any[lo:i_leave + 1] & lit[lo:i_leave + 1]
        during = present[a:b + 1] & lit[a:b + 1]
        if not before.any() or (lit[a:b + 1].any() and not during.any()):
            continue
        t0 = float(sc.t[i_leave])
        t_leave = float(sc.t[a])
        dur = (b - a + 1) * DT
        cash_seen = bool(cash[lo:b + 1].any())
        if sc.bag_on_desk[lo:i_leave + 1].any():
            what = "остаётся инкассаторская сумка"
        elif cash_seen:
            what = "остаётся пачка купюр (светлый прямоугольный предмет на столешнице)"
        else:
            what = "остаются наличные/ценности (предмет, положенный на столешницу)"
        gone = not sc.cashier_inside[a:b + 1].any()
        where = "вышел из помещения" if gone else "отошёл от стола"
        conf = min(0.95, 0.7 + 0.05 * min(dur, 4.0) + (0.05 if gone else 0.0))
        desc = (
            f"В {t0:.1f} с кассир {('встал и ' if i_leave < a else '')}{where} (с {t_leave:.1f} с вне рабочего места "
            f"не менее {dur:.1f} с), при этом на столешнице без присмотра {what}. "
            f"Ранее принятые наличные и ценности должны храниться в запираемых устройствах, а кассир — не покидать "
            f"рабочее место, оставляя их открыто."
        )
        box = (_cash_box(sc, i_leave) or _cash_box(sc, a)) if cash_seen and not sc.bag_on_desk[lo:i_leave + 1].any() else None
        box = box or _obj_box(sc, i_leave)
        events.append(Event(
            type_code="T1", timecode_sec=round(t0, 2), end_sec=round(float(sc.t[b]), 2), confidence=round(conf, 2),
            description=desc, evidence_times=_ev_times(sc, t_leave, (-0.5, 1.0, 3.0)),
            evidence_captions=["кассир покидает рабочее место, ценности на столе", "ценности на столе без присмотра", "рабочее место без кассира"],
            extras={"evidence_boxes": [box] * 3 if box else [], "away_sec": dur, "gone": gone},
        ))
        break
    return events


# --- T2 -------------------------------------------------------------------

def rule_t2(ctx: VideoContext) -> list[Event]:
    """Сейф оставлен открытым: дверца видна открытой (кассир не у сейфа) не менее T2_OPEN_SEC."""
    sc = _scene(ctx)
    if sc is None or sc.anchors.get("safe") is None or not T2_ENABLED:
        return []
    score = np.asarray([v if v is not None else 0.0 for v in sc.safe_open_score], dtype=float)
    known = np.asarray([v is not None for v in sc.safe_open_score])
    near = sc.cashier_near_safe if len(sc.cashier_near_safe) == len(sc.t) else np.zeros(len(sc.t), bool)
    cond = known & (score >= T2_SCORE) & ~sc.lights_off & ~near
    events: list[Event] = []
    for a, b in sc.runs(cond, T2_OPEN_SEC, fill_gap_sec=0.5):
        t0 = float(sc.t[a])
        dur = (b - a + 1) * DT
        gone = not sc.cashier_inside[a:b + 1].all()
        conf = min(0.92, 0.65 + 0.05 * min(dur, 4.0) + (0.05 if gone else 0.0))
        desc = (
            f"С {t0:.1f} с дверца сейфа (хранилища наличных) видна открытой не менее {dur:.1f} с, при этом кассир "
            f"{'покидает помещение' if gone else 'находится в стороне от сейфа'}; хранилище остаётся открытым без контроля."
        )
        sb = tuple(int(v) for v in sc.anchors["safe"]["bbox"])
        ex = (max(0, int(sb[0] - 0.8 * (sb[2] - sb[0]))), sb[1], sb[2], sb[3])
        events.append(Event(
            type_code="T2", timecode_sec=round(t0, 2), end_sec=round(float(sc.t[b]), 2), confidence=round(conf, 2),
            description=desc, evidence_times=_ev_times(sc, t0, (0.0, 1.0, max(1.5, dur - 0.5))),
            evidence_captions=["дверца сейфа открыта", "сейф открыт, кассир не у сейфа", "сейф по-прежнему открыт"],
            extras={"evidence_boxes": [ex] * 3, "open_sec": dur},
        ))
        break
    return events


# --- T3 (при включённом свете) ---------------------------------------------

def rule_t3_lit(ctx: VideoContext) -> list[Event]:
    """Экран АРМ активен, кассир ушёл с рабочего места (при освещении; тёмный вариант — detectors.T3)."""
    sc = _scene(ctx)
    if sc is None or len(sc.t) == 0:
        return []
    active = np.asarray([s == "active" for s in sc.screen])
    if active.sum() < 2:
        return []
    events: list[Event] = []
    for i_leave, a, b in _leave_runs(sc):
        seg = active[a:b + 1]
        if seg.sum() * DT < T3_AWAY_SEC or not active[:a].any():
            continue
        t0 = float(sc.t[i_leave])
        dur = (b - a + 1) * DT
        gone = not sc.cashier_inside[a:b + 1].any()
        conf = min(0.93, 0.7 + 0.04 * min(dur, 5.0) + (0.05 if gone else 0.0))
        desc = (
            f"В {t0:.1f} с кассир {'встал и ' if i_leave < a else ''}{'вышел из помещения' if gone else 'отошёл от рабочего места'}, "
            f"экран ноутбука (АРМ) остаётся активным не менее {seg.sum() * DT:.1f} с (яркий рабочий экран, не экран блокировки). "
            f"Рабочее место при уходе не заблокировано."
        )
        lap = sc.anchors.get("laptop")
        box = tuple(int(v) for v in lap["bbox"]) if lap else None
        events.append(Event(
            type_code="T3", timecode_sec=round(t0, 2), end_sec=round(float(sc.t[b]), 2), confidence=round(conf, 2),
            description=desc, evidence_times=_ev_times(sc, float(sc.t[a]), (-0.5, 1.5, 3.0)),
            evidence_captions=["кассир покидает рабочее место, экран активен", "экран активен без кассира", "рабочее место не заблокировано"],
            extras={"evidence_boxes": [box] * 3 if box else []},
        ))
        break
    return events


# --- T4 -------------------------------------------------------------------

def rule_t4(ctx: VideoContext) -> list[Event]:
    """Посторонний в кассовом помещении: одновременно ≥ 3 уверенных человека внутри (кассир + клиент + третий)."""
    sc = _scene(ctx)
    if sc is None or len(sc.t) == 0:
        return []
    cond = (sc.n_strong >= 3) | (sc.n_visitors >= 2)
    events: list[Event] = []
    for a, b in sc.runs(cond, T4_SEC):
        t0 = float(sc.t[a])
        dur = (b - a + 1) * DT
        conf = min(0.92, 0.7 + 0.05 * min(dur, 4.0))
        desc = (
            f"В {t0:.1f} с в кассовом помещении одновременно находятся {int(sc.n_strong[a])} человека не менее {dur:.1f} с: "
            f"помимо кассира и обслуживаемого клиента внутри присутствует ещё одно лицо. Доступ посторонних в кассовый "
            f"узел должен быть исключён."
        )
        events.append(Event(
            type_code="T4", timecode_sec=round(t0, 2), end_sec=round(float(sc.t[b]), 2), confidence=round(conf, 2),
            description=desc, evidence_times=_ev_times(sc, t0, (0.0, 1.0, 2.5)),
            evidence_captions=["третье лицо входит в кассу", "посторонний внутри кассового помещения", "посторонний остаётся внутри"],
            extras={"n": int(sc.n_strong[a])},
        ))
        break
    return events


# --- T5 -------------------------------------------------------------------

def rule_t5(ctx: VideoContext) -> list[Event]:
    """Посетитель в рабочей зоне кассира (за столом, у сейфа/оборудования)."""
    sc = _scene(ctx)
    if sc is None or sc.work_side is None:
        return []
    cond = sc.visitor_in_work_zone
    events: list[Event] = []
    for a, b in sc.runs(cond, T5_SEC):
        t0 = float(sc.t[a])
        dur = (b - a + 1) * DT
        conf = min(0.9, 0.68 + 0.05 * min(dur, 4.0))
        desc = (
            f"В {t0:.1f} с посетитель заходит в рабочую зону кассира (сторона стола с оборудованием и хранилищем) и "
            f"находится там не менее {dur:.1f} с. Клиент должен оставаться по свою сторону кассы; передача наличных — "
            f"через передаточное устройство."
        )
        events.append(Event(
            type_code="T5", timecode_sec=round(t0, 2), end_sec=round(float(sc.t[b]), 2), confidence=round(conf, 2),
            description=desc, evidence_times=_ev_times(sc, t0, (0.0, 1.0, 2.5)),
            evidence_captions=["посетитель переходит в рабочую зону", "посетитель в рабочей зоне кассира", "посетитель остаётся в рабочей зоне"],
        ))
        break
    return events


# --- T6 -------------------------------------------------------------------

def _visitor_visits(sc: Scene) -> list[tuple[int, int, int]]:
    """Посещения: (track_id, i_enter, i_exit) по индексам сетки, где посетитель виден внутри."""
    seen: dict[int, list[int]] = {}
    for i, ids in enumerate(sc.visitor_ids):
        for tid in ids:
            seen.setdefault(tid, []).append(i)
    out = []
    for tid, idxs in seen.items():
        out.append((tid, idxs[0], idxs[-1]))
    return sorted(out, key=lambda v: v[1])


def rule_t6(ctx: VideoContext) -> list[Event]:
    """Пересчёт не в присутствии клиента: клиент покинул кассу, а его наличные остаются у кассира на столе."""
    sc = _scene(ctx)
    if sc is None or len(sc.t) == 0 or not T6_ENABLED:
        return []
    present = _obj_present(sc)
    n = len(sc.t)
    events: list[Event] = []
    for tid, i_in, i_out in _visitor_visits(sc):
        if i_out >= n - 2:          # посетитель не уходил до конца ролика
            continue
        # деньги появились во время визита (не лежали до его прихода)
        before = present[max(0, i_in - 3):i_in + 1]
        if before.any():
            continue
        if not present[i_in:i_out + 1].any():
            continue
        hi = min(n, i_out + 1 + int(T6_AFTER_SEC * GRID_FPS))
        win = present[i_out + 1:hi] & sc.cashier_at_desk[i_out + 1:hi]
        if len(win) == 0 or win.mean() < T6_MIN_FRAC:
            continue
        t0 = float(sc.t[i_out + 1]) if i_out + 1 < n else float(sc.t[i_out])
        desc = (
            f"В {t0:.1f} с клиент покидает кассу, при этом переданные им наличные остаются на столе кассира и "
            f"операция с ними продолжается без клиента. Приём и пересчёт наличных должны выполняться в присутствии клиента."
        )
        box = _obj_box(sc, min(n - 1, i_out + 2))
        events.append(Event(
            type_code="T6", timecode_sec=round(t0, 2), end_sec=round(float(sc.t[hi - 1]), 2), confidence=0.72,
            description=desc, evidence_times=_ev_times(sc, t0, (-1.0, 0.5, 2.5)),
            evidence_captions=["клиент уходит от кассы", "наличные остаются у кассира после ухода клиента", "операция продолжается без клиента"],
            extras={"evidence_boxes": [box] * 3 if box else [], "track": tid},
        ))
        break
    return events


# --- T7 -------------------------------------------------------------------

def rule_t7(ctx: VideoContext) -> list[Event]:
    """Наличные лежат открыто на столе в момент входа посетителя."""
    sc = _scene(ctx)
    if sc is None or len(sc.t) == 0:
        return []
    def cashlike(o):
        h, sat, v = o["hsv"]
        if o["area"] < T7_MIN_AREA or v < T7_MIN_VAL:
            return False
        return sat >= T7_MIN_SAT or (35 <= h <= 140 and sat >= 15)
    objs = np.asarray([any(cashlike(o) for o in objs_i) for objs_i in sc.desk_objs], dtype=bool)
    cash = _cash(sc, fill_gap_sec=T7_CASH_GAP_SEC)
    objs = objs | cash
    k = int(T7_BEFORE_SEC * GRID_FPS)
    events: list[Event] = []
    for tid, i_in, i_out in _visitor_visits(sc):
        if i_in < k or (i_out - i_in + 1) * DT < 2.0:
            continue
        if not objs[i_in - k:i_in].all():
            continue
        if not sc.cashier_at_desk[i_in - k:i_in].any():
            continue
        t0 = float(sc.t[i_in])
        by_cash = bool(cash[i_in - k:i_in].all())
        what = (f"лежит пачка купюр (светлый прямоугольный предмет, виден не менее {T7_BEFORE_SEC:.1f} с до входа)" if by_cash
                else f"лежат наличные/ценности (видны не менее {T7_BEFORE_SEC:.1f} с до входа)")
        desc = (
            f"В {t0:.1f} с в кассу входит посетитель, при этом на столешнице кассира открыто {what}. "
            f"Ранее принятые деньги должны находиться в запираемых "
            f"устройствах, а не открыто на столе при приёме следующего клиента."
        )
        box = _cash_box(sc, i_in - 1) if by_cash else None
        box = box or _obj_box(sc, i_in - 1)
        events.append(Event(
            type_code="T7", timecode_sec=round(t0, 2), confidence=0.74, description=desc,
            evidence_times=_ev_times(sc, t0, (-1.0, 0.0, 1.5)),
            evidence_captions=["наличные открыто на столе до входа посетителя", "посетитель входит, наличные на столе", "наличные остаются открыто"],
            extras={"evidence_boxes": [box] * 3 if box else [], "track": tid},
        ))
        break
    return events


# --- T8 -------------------------------------------------------------------

def rule_t8(ctx: VideoContext) -> list[Event]:
    """Личные вещи, еда, напитки на рабочем месте: стакан/бутылка или сумка/рюкзак в зоне стола."""
    sc = _scene(ctx)
    if sc is None or len(sc.t) == 0:
        return []
    zone = _desk_zone(sc)
    if zone is None:
        return []
    early = [d for i in range(min(len(sc.t), int(1.0 * GRID_FPS) + 1)) for d in sc.items[i]]
    drink, bag = [], []
    for i in range(len(sc.t)):
        for d in sc.items[i]:
            if not _overlap(d["xyxy"], zone):
                continue
            if d["cls"] in ("cup", "bottle", "wine glass"):
                drink.append((i, d))
            elif d["cls"] in ("handbag", "backpack", "suitcase") and sc.t[i] >= 1.0:
                if any(_iou(d["xyxy"], e["xyxy"]) >= 0.5 for e in early):
                    continue   # статичный предмет интерьера, который детектор путает с сумкой
                bag.append((i, d))
    events: list[Event] = []
    if len(drink) >= T8_DRINK_MIN_DET and max(d["score"] for _, d in drink) >= T8_DRINK_MIN_SCORE:
        i0, d0 = drink[0]
        t0 = float(sc.t[i0])
        best = max(drink, key=lambda x: x[1]["score"])[1]
        desc = (
            f"С {t0:.1f} с на рабочем столе кассира находится стакан/бутылка (напиток), рядом с наличными и "
            f"оборудованием; кассир пользуется им на рабочем месте. Личные вещи, еда и напитки на кассовом рабочем "
            f"месте не допускаются."
        )
        box = tuple(int(v) for v in best["xyxy"])
        events.append(Event(
            type_code="T8", timecode_sec=round(t0, 2), confidence=0.72, description=desc,
            evidence_times=_ev_times(sc, float(sc.t[max(drink, key=lambda x: x[1]['score'])[0]]), (0.0, 1.0, 2.0)),
            evidence_captions=["напиток на рабочем столе кассира", "напиток рядом с наличными", "напиток на рабочем месте"],
            extras={"evidence_boxes": [box] * 3, "kind": "drink", "n_det": len(drink)},
        ))
    elif False and len(bag) >= T8_BAG_MIN_DET and max(d["score"] for _, d in bag) >= T8_BAG_MIN_SCORE:   # сумки по YOLO: ложные на интерьере — выключено
        i0, d0 = bag[0]
        t0 = float(sc.t[i0])
        best = max(bag, key=lambda x: x[1]["score"])[1]
        desc = (
            f"С {t0:.1f} с кассир приносит в кассу личную сумку и держит её на рабочем столе. Личные вещи работников "
            f"должны храниться вне кассового узла."
        )
        box = tuple(int(v) for v in best["xyxy"])
        events.append(Event(
            type_code="T8", timecode_sec=round(t0, 2), confidence=0.66, description=desc,
            evidence_times=_ev_times(sc, t0, (0.0, 1.5, 3.0)),
            evidence_captions=["личная сумка в кассе", "личная сумка на рабочем столе", "сумка остаётся на рабочем месте"],
            extras={"evidence_boxes": [box] * 3, "kind": "bag", "n_det": len(bag)},
        ))
    return events


# --- T10 ------------------------------------------------------------------

def rule_t10(ctx: VideoContext) -> list[Event]:
    """Инкассаторская сумка попадает на стол кассира сразу после входа посетителя (≤ T10_WINDOW_SEC):
    передача внутри кассы из рук в руки без предъявления и проверки документов (ср. X-16: сначала документы, потом сумка)."""
    sc = _scene(ctx)
    if sc is None or len(sc.t) == 0:
        return []
    bag = sc.bag_on_desk
    k = int(1.0 * GRID_FPS)
    events: list[Event] = []
    visits = _visitor_visits(sc)
    for i in range(k, len(sc.t)):
        if not (bag[i] and not bag[max(0, i - k):i].any() and bag[i:min(len(sc.t), i + 3)].all()):
            continue
        t_bag = float(sc.t[i])
        recent = [v for v in visits if 0.0 <= t_bag - float(sc.t[v[1]]) <= T10_WINDOW_SEC and v[2] >= i - 1]
        if not recent:
            continue
        t_in = float(sc.t[recent[0][1]])
        desc = (
            f"В {t_in:.1f} с в кассовое помещение входит посетитель (инкассатор) и уже в {t_bag:.1f} с, через "
            f"{t_bag - t_in:.1f} с, инкассаторская сумка оказывается на столе кассира: передача ценностей происходит "
            f"внутри кассы из рук в руки, без предъявления документов и проверки целостности сумки и пломбы, а "
            f"посторонний находится у рабочего места кассира."
        )
        events.append(Event(
            type_code="T10", timecode_sec=round(t_bag, 2), end_sec=None, confidence=0.72, description=desc,
            evidence_times=_ev_times(sc, t_in, (0.0, max(0.5, (t_bag - t_in) / 2), t_bag - t_in + 1.0)),
            evidence_captions=["посетитель входит в кассу", "посетитель у стола кассира с сумкой", "сумка на столе кассира"],
        ))
        break
    return events


# --- T13 ------------------------------------------------------------------

def rule_t13(ctx: VideoContext) -> list[Event]:
    """Касса без кассира: в начале записи помещение пусто (дверь открыта/не заперта), затем кто-то входит."""
    sc = _scene(ctx)
    if sc is None or len(sc.t) == 0 or not T13_ENABLED:
        return []
    empty = sc.n_inside == 0
    runs = sc.runs(empty, T13_EMPTY_SEC)
    if not runs or runs[0][0] != 0:
        return []
    a, b = runs[0]
    if b >= len(sc.t) - 1:
        return []   # пусто весь ролик — не о чем говорить
    t0 = float(sc.t[a])
    dur = (b - a + 1) * DT
    desc = (
        f"С начала записи ({t0:.1f} с) кассовое помещение остаётся без кассира не менее {dur:.1f} с, затем в {float(sc.t[b + 1]):.1f} с "
        f"в него входят. Касса с оборудованием и хранилищем не должна оставаться без кассира с незапертой дверью."
    )
    return [Event(
        type_code="T13", timecode_sec=round(t0, 2), end_sec=round(float(sc.t[b]), 2), confidence=0.58, description=desc,
        evidence_times=_ev_times(sc, t0, (0.0, max(0.5, dur / 2), dur + 0.5)),
        evidence_captions=["касса без кассира", "помещение пусто", "в кассу входят"],
    )]
