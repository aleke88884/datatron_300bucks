"""Сцена: роли людей, зоны и таблица состояний на общей временной сетке (2 к/с).

Вход — Perception (детекции/треки, 3 к/с) и словарь states (якоря сцены и признаки по кадрам, 2 к/с).
Выход — Scene: для каждого момента сетки булевы/числовые состояния, по которым работают правила T1–T13:

    cashier_inside, cashier_at_desk, cashier_h (высота рамки кассира),
    n_inside (все «внутри»), n_strong (внутри, уверенные детекции), n_visitors, visitor_in_work_zone,
    visitor_ids (кто из посетителей внутри), screen ('active'|'locked'|'off'|None), lights_off,
    bag_on_desk (зелёная инкассаторская сумка), desk_obj_area (статичные предметы на столешнице),
    safe_open_score (0..1, None если сейф не найден/перекрыт).

Роли — только по положению и движению (лица не используются):
  * «кассир» — трек, который дольше всех находится в рабочей зоне (половина плоскости по сторону
    стола, где стоит ноутбук/сейф; если их нет — сторона, где был человек в начале ролика);
  * «посетитель» — уверенный трек внутри кассы, не кассир;
  * «фантом» — слабые/мелкие детекции (люди за стеклом, отражения) — в подсчётах не участвуют.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .camera import REF_FRAME_H
from .perception import Perception, PersonObs, Track
from .video import VideoInfo

GRID_FPS = 2.0
STRONG_SCORE = 0.62          # уверенная детекция человека
STRONG_H = 0.10              # минимальная высота рамки (доля высоты эталонного кадра 848, × масштаб сцены) для «настоящего» человека внутри
CASHIER_MIN_WORK_FRAC = 0.5  # доля наблюдений трека в рабочей зоне, чтобы считать его кассиром
DESK_PAD = 0.12              # расширение bbox стола для теста «у стола»
# Площади в Scene (bag_area, desk_obj_area, desk_objs[].area) приведены к пикселям эталонного кадра 464x848
# делением на масштаб сцены² (camera.scene_scale): пороги ниже не зависят от разрешения и приближения камеры.
BAG_AREA_MIN = 60            # зелёная сумка на столе (эталонные пикс.)
DESK_OBJ_MIN = 150           # статичный предмет на столешнице (эталонные пикс.)
SCREEN_ACTIVE = dict(mean=125.0, p90=185.0, bmr=10.0)   # «экран активен»: яркий, светлый, холодный оттенок (X-28)
SCREEN_LOCKED = dict(std=50.0, p90=180.0)                # «экран заблокирован»: яркие обои, высокая дисперсия (X-29, X-22)
SAFE_OPEN_DELTA = 0.20        # прирост доли тёмных пикселей относительно базовой (10-й перцентиль) → «открыт»
LIGHTS_OFF_RATIO = 0.75       # автоэкспозиция камеры после выключения света восстанавливает яркость до ~0.66 медианы


@dataclass
class Scene:
    info: VideoInfo
    t: np.ndarray                               # сетка времени
    cashier_inside: np.ndarray                  # bool
    cashier_at_desk: np.ndarray                 # bool
    cashier_h: np.ndarray                       # float (0 если кассира нет в кадре)
    cashier_center: list[Optional[tuple[float, float]]]
    n_inside: np.ndarray                        # int
    n_strong: np.ndarray                        # int
    n_visitors: np.ndarray                      # int
    visitor_in_work_zone: np.ndarray            # bool
    visitor_ids: list[list[int]]
    screen: list[Optional[str]]
    lights_off: np.ndarray                      # bool
    bag_on_desk: np.ndarray                     # bool
    bag_area: np.ndarray                        # float
    desk_obj_area: np.ndarray                   # float (статичные предметы на столешнице, без людей)
    desk_objs: list[list[dict]]
    safe_open_score: list[Optional[float]]
    roles: dict[int, str]
    anchors: dict
    work_side: Optional[tuple[float, float, float]] = None   # (nx, ny, c): nx*x+ny*y+c>0 — рабочая сторона
    cashier_near_safe: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    desk_obj_area_any: np.ndarray = field(default_factory=lambda: np.zeros(0, float))
    cashier_track_ids: list[int] = field(default_factory=list)
    visitor_track_ids: list[int] = field(default_factory=list)
    items: list[list[dict]] = field(default_factory=list)    # полнокадровые детекции предметов (handbag, cup, ...)
    scale: float = 1.0                                       # масштаб сцены относительно эталонного кадра (camera.py)
    camera: dict = field(default_factory=dict)               # сводка самокалибровки камеры (camera.calibrate)
    notes: list[str] = field(default_factory=list)
    # наличные, сливающиеся с деревом (pipeline/cashdetect.py): пачка бежевых купюр лежит на столе
    cash_present: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    cash_boxes: list = field(default_factory=list)
    cash_area: np.ndarray = field(default_factory=lambda: np.zeros(0, float))

    # --- удобные запросы -------------------------------------------------
    def idx(self, t: float) -> int:
        return int(np.argmin(np.abs(self.t - t)))

    def runs(self, mask: np.ndarray, min_sec: float = 0.0, fill_gap_sec: float = 0.0) -> list[tuple[int, int]]:
        """Непрерывные участки mask==True → [(i0, i1)] включительно, длиной ≥ min_sec; разрывы ≤ fill_gap_sec заполняются."""
        mask = np.asarray(mask, bool).copy()
        g = int(round(fill_gap_sec * GRID_FPS))
        if g > 0:
            idx = np.flatnonzero(mask)
            for a, b in zip(idx[:-1], idx[1:]):
                if 1 < b - a <= g + 1:
                    mask[a:b] = True
        out: list[list[int]] = []
        for i, v in enumerate(mask):
            if not v:
                continue
            if out and out[-1][1] == i - 1:
                out[-1][1] = i
            else:
                out.append([i, i])
        dt = 1.0 / GRID_FPS
        return [(a, b) for a, b in out if (b - a + 1) * dt >= min_sec]

    def stood_up_before(self, i: int, window_sec: float = 8.0, ratio: float = 1.2) -> Optional[int]:
        """Индекс момента, когда рамка кассира резко выросла (встал) в окне до i; None, если не было."""
        lo = max(0, i - int(window_sec * GRID_FPS))
        pre = self.cashier_h[max(0, lo - int(6 * GRID_FPS)):lo]
        pre = pre[pre > 0]
        win = self.cashier_h[lo:i + 1]
        if len(pre) >= 3:
            base = float(np.median(pre))
        else:
            w = win[win > 0]
            if len(w) < 3:
                return None
            base = float(np.min(w[: max(2, len(w) // 3)]))
        if base <= 0:
            return None
        for k in range(lo, i + 1):
            if self.cashier_h[k] >= ratio * base:
                return k
        return None


# ---------------------------------------------------------------------------
# геометрия
# ---------------------------------------------------------------------------

def _center(xyxy) -> tuple[float, float]:
    return ((xyxy[0] + xyxy[2]) / 2.0, (xyxy[1] + xyxy[3]) / 2.0)


def _overlap(a, b, pad=0.0) -> bool:
    return a[0] - pad < b[2] and a[2] + pad > b[0] and a[1] - pad < b[3] and a[3] + pad > b[1]


def _desk_axis(desk: dict) -> tuple[tuple[float, float], tuple[float, float]]:
    """Центр стола и единичная нормаль к его длинной оси."""
    cx, cy = desk["center"]
    ax, ay = desk["axis"]
    n = np.hypot(ax, ay) or 1.0
    ax, ay = ax / n, ay / n
    return (cx, cy), (-ay, ax)


def _side_value(pt, center, normal) -> float:
    return (pt[0] - center[0]) * normal[0] + (pt[1] - center[1]) * normal[1]


def _strong(o: PersonObs, info: VideoInfo, scale: Optional[float] = None) -> bool:
    s = scale if scale else info.height / REF_FRAME_H
    return o.score >= STRONG_SCORE and o.h >= STRONG_H * REF_FRAME_H * s


FRAGMENT_MAX_SEC = 3.0       # обрывок трека кассира: короткий трек «посетителя»…
FRAGMENT_GAP_SEC = 2.0       # …вплотную по времени к треку кассира (без пересечения)…
FRAGMENT_DIST = 1.0          # …и в том же месте (центры рамок ближе роста человека)


def _rejoin_cashier_fragments(perc: Perception, roles: dict[int, str], cashier_ids: list[int]) -> None:
    """Обрывки трека кассира → кассир. При другом ракурсе/повороте кадра детектор чаще теряет человека на 1–2
    кадра, и трек кассира рвётся: короткий кусок (≤ 3 с) становится «посетителем» и может дать ложное
    «посетитель в рабочей зоне». Человек не может исчезнуть в точке, где через ≤ 2 с появляется кассир, —
    такой кусок присоединяется к кассиру. Треки, пересекающиеся с кассиром по времени, не трогаются."""
    if not cashier_ids:
        return
    by_id = {tr.id: tr for tr in perc.tracks}
    changed = True
    while changed:
        changed = False
        for tid, role in list(roles.items()):
            tr = by_id.get(tid)
            if role != "visitor" or tr is None or not tr.obs or tr.t1 - tr.t0 > FRAGMENT_MAX_SEC:
                continue
            for cid in list(cashier_ids):
                ct = by_id.get(cid)
                if ct is None or not ct.obs or (tr.t0 <= ct.t1 and ct.t0 <= tr.t1):
                    continue
                if 0.0 <= ct.t0 - tr.t1 <= FRAGMENT_GAP_SEC:
                    a, b = tr.obs[-1], ct.obs[0]
                elif 0.0 <= tr.t0 - ct.t1 <= FRAGMENT_GAP_SEC:
                    a, b = ct.obs[-1], tr.obs[0]
                else:
                    continue
                (ax, ay), (bx, by) = _center(a.xyxy), _center(b.xyxy)
                if np.hypot(ax - bx, ay - by) <= FRAGMENT_DIST * max(a.h, b.h, 1.0):
                    roles[tid] = "cashier"
                    cashier_ids.append(tid)
                    changed = True
                    break


# ---------------------------------------------------------------------------
# построение сцены
# ---------------------------------------------------------------------------

def build_scene(perc: Perception, states: dict, info: VideoInfo, cash: Optional[dict] = None) -> Scene:
    sc = states.get("scene", {})
    desk = sc.get("desk")
    safe = sc.get("safe")
    laptop = sc.get("laptop")
    frames = states.get("frames", [])
    H, W = info.height, info.width
    notes: list[str] = []
    scale = float(sc.get("scale") or H / REF_FRAME_H)       # масштаб сцены (camera.calibrate), иначе H/848
    area_k = 1.0 / (scale * scale)                            # пикс. кадра → эталонные пикс.
    gain = float(sc.get("exposure_gain") or 1.0)             # приведение яркости экрана к эталонной экспозиции

    # --- сетка времени: по кадрам состояний (2 к/с), иначе по длительности
    if frames:
        t = np.asarray([f["t"] for f in frames], dtype=float)
    else:
        t = np.arange(0.0, max(info.duration_sec, 0.5), 1.0 / GRID_FPS)
    n = len(t)

    # --- геометрия стола
    center = normal = None
    desk_bbox = None
    if desk is not None:
        center, normal = _desk_axis(desk)
        desk_bbox = tuple(desk["bbox"])
    else:
        notes.append("стол не найден — зоны не построены, роли по присутствию в начале ролика")

    def desk_overlap(xyxy) -> bool:
        return desk_bbox is None or _overlap(xyxy, desk_bbox)

    # --- роли треков: кассир — самый долгий уверенный трек внутри кассы, работающий у стола;
    #     остальные уверенные треки у стола, не пересекающиеся с кассиром по времени, — тоже кассир (ушёл/вернулся);
    #     прочие уверенные — посетители; слабые/мелкие — фантомы (за стеклом, отражения).
    roles: dict[int, str] = {}
    stats: dict[int, dict] = {}
    for tr in perc.tracks:
        inside_obs = [o for o in tr.obs if o.inside]
        strong_obs = [o for o in inside_obs if _strong(o, info, scale)]
        if not inside_obs:
            roles[tr.id] = "outside"
            continue
        if len(strong_obs) < max(2, 0.5 * len(inside_obs)):
            roles[tr.id] = "phantom"
            continue
        roles[tr.id] = "visitor"
        at_desk_n = sum(1 for o in strong_obs if desk_overlap(o.xyxy))
        stats[tr.id] = {"n": len(strong_obs), "desk": at_desk_n, "t0": tr.t0, "t1": tr.t1,
                        "side": float(np.median([_side_value(_center(o.xyxy), center, normal) for o in strong_obs])) if center else 0.0}
    cashier_ids: list[int] = []
    if stats:
        main = max(stats, key=lambda k: stats[k]["desk"])
        if stats[main]["desk"] >= 2:
            roles[main] = "cashier"
            cashier_ids.append(main)
            m0, m1 = stats[main]["t0"], stats[main]["t1"]
            for tid, st_ in stats.items():
                if tid == main or st_["desk"] < max(3, 0.5 * st_["n"]):
                    continue
                ov = max(0.0, min(m1, st_["t1"]) - max(m0, st_["t0"]))
                if ov <= 0.2 * max(0.5, st_["t1"] - st_["t0"]):
                    roles[tid] = "cashier"
                    cashier_ids.append(tid)
        else:
            notes.append("нет трека, работающего у стола — кассир не определён")
    _rejoin_cashier_fragments(perc, roles, cashier_ids)
    # рабочая сторона стола — та, где находится кассир
    work_side = None
    if center is not None and cashier_ids:
        side_med = float(np.median([stats[t]["side"] for t in cashier_ids]))
        sign = 1.0 if side_med >= 0 else -1.0
        work_side = (normal[0] * sign, normal[1] * sign, -(center[0] * normal[0] + center[1] * normal[1]) * sign)

    def in_work_zone(xyxy) -> Optional[bool]:
        if work_side is None:
            return None
        cx, cy = _center(xyxy)
        return (work_side[0] * cx + work_side[1] * cy + work_side[2]) > 0

    visitor_ids_all = [tid for tid, r in roles.items() if r == "visitor"]

    # --- по сетке: люди
    desk_box = None
    foot_box = None
    if desk is not None:
        x1, y1, x2, y2 = desk["bbox"]
        w, h = (x2 - x1), (y2 - y1)
        desk_box = (x1 - w * DESK_PAD, y1 - h * DESK_PAD, x2 + w * DESK_PAD, y2 + h * DESK_PAD)
        foot_box = (x1 - 0.25 * w, y1 - 0.35 * h, x2 + 0.25 * w, y2 + 1.3 * h)   # нижняя точка рамки кассира
    safe_ext = None
    if safe is not None:
        sx1, sy1, sx2, sy2 = safe["bbox"]
        safe_ext = (sx1 - 0.8 * (sx2 - sx1), sy1 - 0.2 * (sy2 - sy1), sx2 + 0.2 * (sx2 - sx1), sy2 + 0.3 * (sy2 - sy1))

    cashier_inside = np.zeros(n, bool)
    cashier_at_desk = np.zeros(n, bool)
    cashier_h = np.zeros(n, float)
    cashier_near_safe = np.zeros(n, bool)
    cashier_center: list[Optional[tuple[float, float]]] = [None] * n
    n_inside = np.zeros(n, int)
    n_strong = np.zeros(n, int)
    n_visitors = np.zeros(n, int)
    visitor_in_work = np.zeros(n, bool)
    visitor_ids: list[list[int]] = [[] for _ in range(n)]
    items: list[list[dict]] = [[] for _ in range(n)]
    ptimes = np.asarray(perc.times) if perc.times else np.zeros(0)
    for i, ti in enumerate(t):
        if len(ptimes) == 0:
            continue
        j = int(np.argmin(np.abs(ptimes - ti)))
        if abs(ptimes[j] - ti) > 0.6:
            continue
        ps = perc.persons[j]
        n_inside[i] = sum(1 for o in ps if o.inside)
        n_strong[i] = sum(1 for o in ps if o.inside and _strong(o, info, scale))
        best_c = None
        for o in ps:
            if not o.inside:
                continue
            role = roles.get(o.track_id, "phantom")
            if role == "cashier":
                cashier_inside[i] = True
                if best_c is None or o.h > best_c.h:
                    best_c = o
            elif role == "visitor" and _strong(o, info, scale):
                n_visitors[i] += 1
                visitor_ids[i].append(o.track_id)
                wz = in_work_zone(o.xyxy)
                if wz:
                    visitor_in_work[i] = True
        if best_c is not None:
            cashier_h[i] = best_c.h
            cashier_center[i] = _center(best_c.xyxy)
            fx, fy = best_c.foot
            cashier_at_desk[i] = desk_box is None or _overlap(best_c.xyxy, desk_box) or (
                foot_box[0] <= fx <= foot_box[2] and foot_box[1] <= fy <= foot_box[3])
            cx_, cy_ = _center(best_c.xyxy)
            cashier_near_safe[i] = safe_ext is not None and (safe_ext[0] <= cx_ <= safe_ext[2] and safe_ext[1] <= cy_ <= safe_ext[3])
        for d in perc.objects[j]:
            if d.cls in ("handbag", "backpack", "suitcase", "cup", "bottle", "wine glass", "cell phone", "book"):
                items[i].append({"cls": d.cls, "score": float(d.score), "xyxy": tuple(float(v) for v in d.xyxy)})

    # --- по сетке: признаки состояний
    screen: list[Optional[str]] = [None] * n
    lights_off = np.zeros(n, bool)
    bag_area = np.zeros(n, float)
    desk_obj_area = np.zeros(n, float)
    desk_obj_area_any = np.zeros(n, float)   # включая предметы, перекрытые рамкой человека (кассир стоит над столом)
    desk_objs: list[list[dict]] = [[] for _ in range(n)]
    safe_scores: list[Optional[float]] = [None] * n
    med_b = float(sc.get("median_brightness") or 0.0)
    raw_dark: list[Optional[float]] = [None] * n
    for i, f in enumerate(frames):
        if med_b > 0 and f.get("brightness", med_b) < LIGHTS_OFF_RATIO * med_b:
            lights_off[i] = True
        s = f.get("screen")
        if s and not s.get("occluded") and s.get("mean") is not None:
            mean_, p90_ = s["mean"] * gain, (s.get("p90") or 0) * gain
            bmr_, std_ = (s.get("blue_minus_red") or 0) * gain, (s.get("std") or 0) * gain
            if mean_ >= SCREEN_ACTIVE["mean"] and p90_ >= SCREEN_ACTIVE["p90"] and bmr_ >= SCREEN_ACTIVE["bmr"]:
                screen[i] = "active"
            elif std_ >= SCREEN_LOCKED["std"] and p90_ >= SCREEN_LOCKED["p90"]:
                screen[i] = "locked"
            else:
                screen[i] = "off"
        d = f.get("desk")
        if d:
            bag_area[i] = float(d.get("green_area") or 0) * area_k
            gb = d.get("green_bbox")
            if gb and desk_bbox is not None and gb[1] > desk_bbox[3]:
                # зелёное пятно целиком ниже нижней кромки стола — не на столешнице (сиденье стула, пол):
                # states.desk_mask расширяет зону стола и вниз, а при другой экспозиции/перекодировке бирюзовое
                # сиденье кассира проходит зелёный порог
                bag_area[i] = 0.0
            statics = [b if area_k == 1.0 else {**b, "area": b["area"] * area_k} for b in d.get("static_objs", [])]
            objs = [b for b in statics if not b.get("person")]
            desk_objs[i] = objs
            desk_obj_area[i] = float(sum(b["area"] for b in objs))
            desk_obj_area_any[i] = float(sum(b["area"] for b in statics))
        sf = f.get("safe")
        if sf and not sf.get("occluded"):
            v = sf.get("dark_left")
            if v is None:
                v = sf.get("dark_face")
            raw_dark[i] = v
    # сейф: «открыт», если доля тёмного выше базовой (10-й перцентиль по неперекрытым кадрам) на SAFE_OPEN_DELTA
    vals = [v for v in raw_dark if v is not None]
    if len(vals) >= 4:
        base = float(np.percentile(vals, 10))
        for i, v in enumerate(raw_dark):
            if v is None:
                continue
            safe_scores[i] = float(np.clip((v - base) / SAFE_OPEN_DELTA, 0.0, 1.5))
    bag_on_desk = bag_area >= BAG_AREA_MIN
    from .cashdetect import cash_on_grid

    cash_present, cash_boxes, cash_area = cash_on_grid(cash, t)

    return Scene(
        info=info, t=t, cashier_inside=cashier_inside, cashier_at_desk=cashier_at_desk, cashier_h=cashier_h,
        cashier_center=cashier_center, n_inside=n_inside, n_strong=n_strong, n_visitors=n_visitors,
        visitor_in_work_zone=visitor_in_work, visitor_ids=visitor_ids, screen=screen, lights_off=lights_off,
        bag_on_desk=bag_on_desk, bag_area=bag_area, desk_obj_area=desk_obj_area, desk_objs=desk_objs,
        safe_open_score=safe_scores, roles=roles, anchors={"desk": desk, "safe": safe, "laptop": laptop},
        cashier_near_safe=cashier_near_safe, desk_obj_area_any=desk_obj_area_any,
        work_side=work_side, cashier_track_ids=cashier_ids, visitor_track_ids=visitor_ids_all, items=items, notes=notes,
        cash_present=cash_present, cash_boxes=cash_boxes, cash_area=cash_area,
        scale=scale,
    )


def scene_summary(scene: Scene) -> dict:
    """Компактная сводка для meta ответа и отладки."""
    def frac(a):
        return float(np.mean(a)) if len(a) else 0.0
    return {
        "roles": {str(k): v for k, v in scene.roles.items()},
        "cashier_tracks": scene.cashier_track_ids,
        "visitor_tracks": scene.visitor_track_ids,
        "cashier_inside_frac": round(frac(scene.cashier_inside), 3),
        "max_strong_inside": int(scene.n_strong.max()) if len(scene.n_strong) else 0,
        "screen_states": {s: int(sum(1 for x in scene.screen if x == s)) for s in ("active", "locked", "off")},
        "bag_frames": int(scene.bag_on_desk.sum()),
        "desk_obj_frames": int((scene.desk_obj_area >= DESK_OBJ_MIN).sum()),
        "safe_open_frames": int(sum(1 for v in scene.safe_open_score if v is not None and v >= 1.0)),
        "lights_off_frames": int(scene.lights_off.sum()),
        "cash_frames": int(scene.cash_present.sum()) if len(scene.cash_present) else 0,
        "camera": scene.camera or None,
        "notes": scene.notes,
    }
