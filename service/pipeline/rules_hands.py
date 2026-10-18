"""Правила по позе рук кассира (слой pipeline.pose): T6 и вспомогательные признаки.

Все величины — относительные: координаты кисти в системе тела кассира (начало — середина плеч, единица —
масштаб тела ≈ ширина плеч), «поза покоя» — медиана положения кисти по самому ролику, длительности — в секундах.
Пиксельных зон и порогов, привязанных к камере, нет. Если поза недоступна (нет модели, кисти не видны) —
правила молчат: «неизвестно» не превращается в «нарушение».

T6 (пересчёт/операция с наличными не в присутствии клиента, Правила № 231 п. 20):
    клиент, сидевший у стола, встаёт и уходит (рамка посетителя резко вырастает относительно его же
    «сидячей» высоты и больше не уменьшается до конца трека), а кассир в это время за столом и его рука
    вынесена далеко от позы покоя (тянется к счётной машинке/оборудованию) не менее T6H_ENGAGED_SEC.
    Физический смысл: операция с деньгами клиента продолжается, когда клиент уже уходит и не может
    наблюдать пересчёт. В чистых роликах (Х-5, Х-19, Х-39, Х-48) после вставания клиента руки кассира
    в позе покоя; в Х-39 кассир после ухода клиента тянется к сейфу (убирает деньги — штатно), поэтому рука
    в зоне якоря сейфа не считается «работой с деньгами».
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .context import Event, VideoContext
from .pose import track_series

# --- пороги (относительные) ----------------------------------------------
STAND_RATIO = 1.35          # рамка посетителя выросла в 1.35 раза относительно сидячей высоты — встал
STAND_KEEP_RATIO = 1.2      # после вставания высота не опускается ниже 1.2 × сидячей (не сел обратно)
SEATED_SEC = 3.0            # до вставания посетитель сидел не менее 3 с (рамка стабильна)
SEATED_CV = 0.15            # «стабильна»: разброс высоты ≤ 15 % от медианы
T6H_WINDOW_SEC = 6.0        # окно после вставания клиента, в котором ищем работу рук кассира
T6H_ENGAGED_DIST = 1.1      # кисть дальше 1.1 масштаба тела от своей позы покоя — рука вынесена к предмету (чистые: ≤ 0.81)
T6H_ENGAGED_SEC = 2.0       # и держится так не менее 2 с подряд (короткие пропуски точек до 0.7 с заполняются)
T6H_GAP_SEC = 0.7
T6H_MIN_REST_FRAMES = 10    # для позы покоя нужно ≥ 10 кадров с видимой кистью
SAFE_PAD = 0.05             # зона сейфа = рамка якоря сейфа + 5 % её размера
T6H_CONFIDENCE = 0.62       # индикатор по позе: ниже, чем у правил по состояниям объектов


def _pose(ctx: VideoContext) -> Optional[dict]:
    p = (getattr(ctx, "cache", None) or {}).get("pose")
    return p if p and p.get("times") else None


def visitor_departures(ctx: VideoContext) -> list[dict]:
    """Моменты, когда сидевший у стола посетитель встаёт и больше не садится (уходит).

    Возвращает [{track_id, t_stand, t_end, seated_h}] по трекам посетителей (роль из scene)."""
    sc = getattr(ctx, "scene", None)
    tracks = getattr(ctx, "tracks", None) or []
    if sc is None:
        return []
    visitors = {tid for tid, r in sc.roles.items() if r == "visitor"}
    out = []
    for tr in tracks:
        if tr.id not in visitors:
            continue
        obs = [o for o in tr.obs if o.inside]
        if len(obs) < 8:
            continue
        t = np.asarray([o.t for o in obs], float)
        h = np.asarray([o.h for o in obs], float)
        found = None
        for k in range(len(obs) - 1, 0, -1):
            pre = (t >= t[k] - SEATED_SEC - 0.5) & (t <= t[k] - 0.3)
            if pre.sum() < max(4, int(SEATED_SEC * 2)):
                continue
            base = float(np.median(h[pre]))
            if base <= 0 or float(np.std(h[pre])) > SEATED_CV * base:
                continue
            if h[k] < STAND_RATIO * base:
                continue
            after = h[k:]
            # не сел обратно: не более одного кадра ниже порога удержания
            if int((after < STAND_KEEP_RATIO * base).sum()) > 1:
                continue
            # первый кадр роста (начало вставания)
            k0 = k
            while k0 - 1 >= 0 and h[k0 - 1] >= 1.1 * base and t[k] - t[k0 - 1] <= 1.5:
                k0 -= 1
            found = {"track_id": tr.id, "t_stand": float(t[k0]), "t_end": float(t[-1]), "seated_h": base}
            break
        if found:
            out.append(found)
    return sorted(out, key=lambda d: d["t_stand"])


def cashier_engagement(ctx: VideoContext) -> Optional[dict]:
    """Ряд «рука кассира вынесена от позы покоя»: t, dist (макс. по кистям, масштабы тела; NaN — не видно)."""
    pose = _pose(ctx)
    sc = getattr(ctx, "scene", None)
    if pose is None or sc is None or not sc.cashier_track_ids:
        return None
    s = track_series(pose, sc.cashier_track_ids)
    rel = s["wrists_rel"]
    n = len(s["t"])
    if n == 0:
        return None
    at_desk = np.zeros(n, bool)
    for i, ti in enumerate(s["t"]):
        j = sc.idx(float(ti))
        at_desk[i] = bool(sc.cashier_at_desk[j]) and abs(float(sc.t[j]) - ti) <= 0.6
    # зона сейфа (якорь сцены, найден по кадрам): рука в сейфе — убирает ценности в хранилище, это штатное
    # действие (Правила № 231 п. 20 — хранение в запираемых устройствах), а не работа с деньгами клиента
    safe_zone = None
    safe = (getattr(sc, "anchors", None) or {}).get("safe")
    if safe and safe.get("bbox"):
        x1, y1, x2, y2 = (float(v) for v in safe["bbox"])
        px_, py_ = SAFE_PAD * (x2 - x1), SAFE_PAD * (y2 - y1)
        safe_zone = (x1 - px_, y1 - py_, x2 + px_, y2 + py_)
    wpx = s["wrist_px"]
    in_safe = np.zeros((n, 2), bool)
    if safe_zone is not None:
        with np.errstate(invalid="ignore"):
            in_safe = ((wpx[:, :, 0] >= safe_zone[0]) & (wpx[:, :, 0] <= safe_zone[2])
                       & (wpx[:, :, 1] >= safe_zone[1]) & (wpx[:, :, 1] <= safe_zone[3]))
    # поза покоя — медиана положения кисти, пока кассир у стола (без кадров, где рука в сейфе)
    rest = np.full((2, 2), np.nan)
    for hnd in range(2):
        ok = at_desk & ~np.isnan(rel[:, hnd, 0]) & ~in_safe[:, hnd]
        if ok.sum() >= T6H_MIN_REST_FRAMES:
            rest[hnd] = np.median(rel[ok, hnd], axis=0)
    dist = np.full(n, np.nan)
    for i in range(n):
        d = []
        for hnd in range(2):
            if np.isnan(rel[i, hnd, 0]) or np.isnan(rest[hnd, 0]):
                continue
            d.append(0.0 if in_safe[i, hnd] else float(np.hypot(*(rel[i, hnd] - rest[hnd]))))
        if d:
            dist[i] = max(d)
    return {"t": s["t"], "dist": dist, "at_desk": at_desk, "rest": rest, "rel": rel, "in_safe": in_safe}


def _engaged_runs(t: np.ndarray, mask: np.ndarray, unknown: np.ndarray, gap_sec: float) -> list[tuple[int, int]]:
    """Участки mask==True; короткие разрывы из «неизвестно» (кисть не видна) до gap_sec заполняются."""
    m = mask.copy()
    idx = np.flatnonzero(m)
    for a, b in zip(idx[:-1], idx[1:]):
        if b - a > 1 and t[b] - t[a] <= gap_sec + 1e-6 and unknown[a + 1:b].all():
            m[a:b] = True
    runs: list[list[int]] = []
    for i, v in enumerate(m):
        if not v:
            continue
        if runs and runs[-1][1] == i - 1:
            runs[-1][1] = i
        else:
            runs.append([i, i])
    return [(a, b) for a, b in runs]


def _track_box(ctx: VideoContext, track_ids: list[int], t: float) -> Optional[tuple[int, int, int, int]]:
    """Рамка человека из указанных треков, ближайшая к моменту t (для подсветки на кадре-доказательстве)."""
    best = None
    ids = set(track_ids)
    for tr in getattr(ctx, "tracks", None) or []:
        if tr.id not in ids:
            continue
        o = tr.at(t, tol=0.7)
        if o is not None and (best is None or abs(o.t - t) < abs(best.t - t)):
            best = o
    return tuple(int(v) for v in best.xyxy) if best is not None else None


def rule_t6_hands(ctx: VideoContext) -> list[Event]:
    """T6 по позе: клиент встал и уходит, а рука кассира вынесена к оборудованию/деньгам ≥ 2 с."""
    eng = cashier_engagement(ctx)
    if eng is None:
        return []
    sc = ctx.scene
    t, dist, at_desk = eng["t"], eng["dist"], eng["at_desk"]
    unknown = np.isnan(dist)
    engaged = (~unknown) & (dist >= T6H_ENGAGED_DIST) & at_desk
    events: list[Event] = []
    for dep in visitor_departures(ctx):
        t_st = dep["t_stand"]
        lo, hi = t_st, t_st + T6H_WINDOW_SEC
        # посетитель действительно уходит: встал не позже чем за 2 с до конца записи
        if t_st > sc.info.duration_sec - 2.0:
            continue
        for a, b in _engaged_runs(t, engaged, unknown, T6H_GAP_SEC):
            if t[a] < lo - 0.01 or t[a] > hi:
                continue
            dur = float(t[b] - t[a]) + 1.0 / 3.0
            if dur < T6H_ENGAGED_SEC:
                continue
            t0 = float(t[a])
            times = [max(0.0, t_st), t0 + min(1.0, dur / 2), min(float(t[b]), sc.info.duration_sec - 0.1)]
            boxes = [_track_box(ctx, [dep["track_id"]], times[0])] + [_track_box(ctx, sc.cashier_track_ids, x) for x in times[1:]]
            desc = (
                f"Признак операции с наличными без клиента (требует проверки): в {t_st:.1f} с клиент встаёт и "
                f"уходит от кассы, а с {t0:.1f} с кассир за столом продолжает работу — рука вынесена к "
                f"оборудованию/наличным не менее {dur:.1f} с, уже без клиента. Что именно делает кассир, по позе "
                f"рук не устанавливается. Может указывать на пересчёт не в присутствии клиента (п. 20 абз. 1 "
                f"Правил № 231: наличные принимаются и пересчитываются в присутствии клиента)."
            )
            events.append(Event(
                type_code="T6", timecode_sec=round(t0, 2), end_sec=round(float(t[b]), 2),
                confidence=T6H_CONFIDENCE, description=desc,
                evidence_times=times,
                evidence_captions=["клиент встаёт и уходит от кассы", "кассир продолжает операцию без клиента",
                                   "операция продолжается после ухода клиента"],
                extras={"source": "pose", "track": dep["track_id"], "t_stand": t_st, "engaged_sec": dur,
                        "evidence_boxes": boxes if all(bx is not None for bx in boxes) else []},
            ))
            break
        if events:
            break
    return events
