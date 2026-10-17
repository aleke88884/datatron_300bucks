"""Наличные на деревянной столешнице: купюры, сливающиеся с цветом стола (5000 тг — бежево-коричневые).

Проблема («криптическая окраска»): оттенок пачки бежевых купюр попадает в диапазон цвета дерева (WOOD_HSV),
поэтому признак «было дерево — стало не дерево» (states.py) пачку не видит. Как у хищников, которые замечают
неподвижную добычу, окрашенную под фон, по контрасту светлоты и краям, а не по цвету, здесь пачка выделяется
относительно СОБСТВЕННОЙ статистики дерева в этом же кадре:

  * константность цвета: баланс белого приводится к нейтральному по самым светлым поверхностям (стены), чтобы
    камера с другим балансом белого видела дерево так же (wb_gains);
  * столешница — устойчиво «деревянные» пиксели по выборке кадров (без лиц/рук под рамками людей), с заливкой дыр
    (desk_support) — не пиксельная зона, а вывод из самого видео;
  * «бледность» пикселя = падение насыщенности (доля S дерева) + прирост светлоты (доля V дерева), сглаженная по
    окрестности: пороги — доли от медианы дерева в этом же кадре, поэтому не зависят от экспозиции (paleness);
  * человеческие рамки маскируются (розовая кофта кассира тоже «бледная»), а предмет под рамкой человека
    не «исчезает», а удерживается, пока его место снова не станет видно (persistence through occlusion);
  * трек «пачка купюр» — тот же тон, что у дерева, бледнее, но не белая бумага (is_cashlike), и лежит на месте
    не менее CASH_STATIC_SEC секунд (cash_tracks / cash_on_grid).

Размеры — доли площади столешницы, время — секунды; пиксельных зон, привязанных к этой камере, нет.

Выход compute_cash_track(...) — только числа (кеш results/cash/<видео>.json):
    {"version", "fps", "desk": {bbox, area}, "wb_gains", "frames": [{"t", "wood_s", "wood_v", "objs": [...]}, ...]}
где objs — бледные предметы на столе (трекинг): id, bbox, area, area_rel (доля столешницы), age_sec, seen (виден
сейчас) / occluded (перекрыт человеком), s_rel/v_rel (насыщенность/светлота относительно дерева), hue/dhue (тон и
его отличие от тона дерева), peak (пик бледности), fill (заполнение рамки), white (доля почти белых пикселей).
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .perception import Perception
from .video import VideoInfo, iter_frames, open_capture

log = logging.getLogger("pipeline.cash")

CASH_VERSION = 1
CASH_FPS = 2.0                 # та же сетка, что у states (2 к/с)
WOOD_HSV = ((8, 40, 120), (28, 140, 235))   # совпадает с states.WOOD_HSV (не импортируем, чтобы не связывать модули)
DESK_PERSIST = 0.4             # пиксель — столешница, если он дерево хотя бы в 40 % выборочных кадров
DESK_MIN_COMP = 0.15           # компоненты дерева меньше 15 % крупнейшей отбрасываются (дверь, плинтус)
DESK_TOP_FRAC = 0.18           # выше 18 % высоты кадра стола быть не может (стекло/потолок)
PALE_V_REL = 0.2               # прирост светлоты нормируется на 20 % V дерева
PALE_SCORE = 0.6               # порог сглаженной «бледности» (0 — дерево; бумага/купюры > 1)
PERSON_SHRINK = 0.12           # рамка человека сжимается на 12 % с каждой стороны при маскировании
OBJ_MIN_REL = 0.0035           # минимальная площадь предмета — 0.35 % площади столешницы
OBJ_MAX_REL = 0.12             # больше 12 % столешницы — не пачка купюр (засветка, лист бумаги А4 и т. п.)
OBJ_MIN_FILL = 0.30            # заполнение рамки (компактный прямоугольный предмет, не полоса блика)
OBJ_MAX_ASPECT = 6.0
MATCH_IOU = 0.25
MATCH_DIST = 0.6               # или центр в пределах 0.6 диагонали предмета
HOLD_SEC = 30.0                # сколько секунд держать предмет, перекрытый человеком
MISS_SEC = 1.0                 # видимое место пусто дольше 1 с — предмет убран
CASH_STATIC_SEC = 1.5          # «лежит»: на одном месте не менее 1.5 с
WB_TOP_PCT = 90                # баланс белого по 10 % самых светлых (непересвеченных) пикселей — стены
WB_MAX_GAIN = 1.4
# сигнатура «бежевая пачка купюр» (5000 тг): тот же тон, что у дерева, заметно бледнее, но не белая бумага
# Пороги взяты с запасом: на 42 обучающих видео даже при одновременном ослаблении всех порогов
# (оттенок ±9, S 25–90 %, белого < 55 %, V +4 %, пик 1.0, голосов 40 %) правила T1/T7 не дают ложных срабатываний —
# защита от ложных — в логике «состояние стола в момент ухода кассира / входа посетителя», а не в узком цвете.
CASH_DHUE = 8.0                # |оттенок − оттенок дерева| ≤ 8 (розовая кофта/кожа — краснее, документы — без тона)
CASH_S_REL = (0.30, 0.85)      # насыщенность 30–85 % от дерева
CASH_V_REL = 0.05              # светлее дерева минимум на 5 % V
CASH_WHITE_MAX = 0.42          # доля почти белых пикселей < 42 % (документы, удостоверения, чеки — белые)
CASH_PEAK = 1.1                # пик сглаженной бледности (блик на волокнах дерева — ниже)
CASH_TRACK_VOTE = 0.5          # трек — «наличные», если ≥ 50 % его наблюдений похожи на пачку купюр
CASH_TRACK_SEEN = 3


def _wood(hsv: np.ndarray) -> np.ndarray:
    return cv2.inRange(hsv, WOOD_HSV[0], WOOD_HSV[1]) > 0


def sample_with_times(path: str, info: VideoInfo, n: int = 25) -> list[tuple[float, np.ndarray]]:
    """n кадров равномерно по ролику: позиционированием (в ~4 раза быстрее полного декодирования),
    при неточном seek контейнера — последовательным чтением."""
    out: list[tuple[float, np.ndarray]] = []
    cap = open_capture(path)
    try:
        for k in range(n):
            fi = int((k + 0.5) * info.frame_count / n)
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, f = cap.read()
            if not ok or f is None:
                out = []
                break
            out.append((fi / max(info.fps, 1e-6), f))
    finally:
        cap.release()
    if len(out) == n:
        return out
    every = max(1, int(round(info.frame_count / n)))
    return [(t, f) for _, t, f in iter_frames(path, every_n=every)][:n]


def wb_gains(samples: list[tuple[float, np.ndarray]]) -> np.ndarray:
    """Константность цвета («белый лист»): белые стены/мебель — самые светлые поверхности кадра; их средний цвет
    приводится к нейтральному. Камера с другим балансом белого (теплее/холоднее) после этого видит дерево и купюры
    так же, как обучающая; для роликов датасета коэффициенты ≈ 1."""
    if not samples:
        return np.ones(3, np.float32)
    acc = []
    for _, fr in samples[:: max(1, len(samples) // 8)]:
        small = cv2.resize(fr, (max(1, fr.shape[1] // 4), max(1, fr.shape[0] // 4)), interpolation=cv2.INTER_AREA)
        px = small.reshape(-1, 3).astype(np.float32)
        lum = px.mean(1)
        hi = px[(lum >= np.percentile(lum, WB_TOP_PCT)) & (px.max(1) < 254)]
        if len(hi) >= 20:
            acc.append(np.median(hi, axis=0))
    if not acc:
        return np.ones(3, np.float32)
    ref = np.median(np.asarray(acc), axis=0)
    g = ref.mean() / np.maximum(ref, 1.0)
    return np.clip(g, 1.0 / WB_MAX_GAIN, WB_MAX_GAIN).astype(np.float32)


def _balance(img: np.ndarray, g: np.ndarray) -> np.ndarray:
    if np.allclose(g, 1.0, atol=0.01):
        return img
    return np.clip(img.astype(np.float32) * g, 0, 255).astype(np.uint8)


def desk_support(samples: list[tuple[float, np.ndarray]], perc: Optional[Perception] = None,
                 gains: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """Маска столешницы (верх и фронт стола) по выборке кадров.

    Пиксель — стол, если он устойчиво «дерево» по всем кадрам (DESK_PERSIST), но не тогда, когда «деревянным» он
    бывает только под рамкой человека (лицо/руки кассира по цвету похожи на дерево: когда человека нет, там стена).
    Затем — компоненты не выше верхнего края крупнейшей (стол — один предмет мебели, голова человека над ним
    отсекается), замыкание и заливка дыр: предметы, лежащие на столе, оказываются внутри маски."""
    if not samples:
        return None
    H, W = samples[0][1].shape[:2]
    wood_all = np.zeros((H, W), np.float32)
    wood_vis = np.zeros((H, W), np.float32)
    vis_n = np.zeros((H, W), np.float32)
    for t, fr in samples:
        w = _wood(cv2.cvtColor(_balance(fr, gains) if gains is not None else fr, cv2.COLOR_BGR2HSV))
        pm = _person_mask((H, W), _persons_at(perc, t), shrink=0.0) if perc is not None else np.zeros((H, W), bool)
        wood_all += w
        wood_vis += w & ~pm
        vis_n += ~pm
    n_s = len(samples)
    seen = vis_n >= 3
    ratio_vis = wood_vis / np.maximum(vis_n, 1)
    m = (wood_all >= DESK_PERSIST * n_s) & ~(seen & (ratio_vis < 0.3))
    m = m.astype(np.uint8)
    m[: int(DESK_TOP_FRAC * H)] = 0
    k = max(3, int(round(H / 170)) | 1)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2 * k + 1, 2 * k + 1), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return None
    areas = st[1:, 4]
    bi = int(np.argmax(areas)) + 1
    big = int(st[bi, 4])
    if big < 0.004 * H * W:
        return None
    top = st[bi, 1] - 2 * k
    bx1, bx2 = st[bi, 0], st[bi, 0] + st[bi, 2]
    sel = np.zeros((H, W), np.uint8)
    for i in range(1, n):
        x, y, w_, h_, a = st[i]
        if a < DESK_MIN_COMP * big or y < top or x > bx2 or x + w_ < bx1:
            continue
        sel[lab == i] = 1
    sel = cv2.morphologyEx(sel, cv2.MORPH_CLOSE, np.ones((6 * k + 1, 6 * k + 1), np.uint8))
    cnts, _ = cv2.findContours(sel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros((H, W), np.uint8)
    cv2.drawContours(out, cnts, -1, 1, thickness=cv2.FILLED)
    return out > 0


def _persons_at(perc: Optional[Perception], t: float, tol: float = 0.5) -> list:
    if perc is None or not perc.times:
        return []
    j = perc.frame_index(t)
    return perc.persons[j] if abs(perc.times[j] - t) <= tol else []


def _person_mask(shape, persons, shrink: float = PERSON_SHRINK) -> np.ndarray:
    H, W = shape
    m = np.zeros((H, W), bool)
    for p in persons:
        a, b, c, d = p.xyxy
        w, h = (c - a) * shrink, (d - b) * shrink
        x1, y1, x2, y2 = int(max(0, a + w)), int(max(0, b + h)), int(min(W, c - w)), int(min(H, d - h))
        if x2 > x1 and y2 > y1:
            m[y1:y2, x1:x2] = True
    return m


def _iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _near(a, b) -> bool:
    if _iou(a, b) >= MATCH_IOU:
        return True
    ca = ((a[0] + a[2]) / 2, (a[1] + a[3]) / 2)
    cb = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    diag = max(np.hypot(a[2] - a[0], a[3] - a[1]), np.hypot(b[2] - b[0], b[3] - b[1]), 1.0)
    return float(np.hypot(ca[0] - cb[0], ca[1] - cb[1])) <= MATCH_DIST * diag


def _hue_diff(h: float, ref: float) -> float:
    """Разность оттенков OpenCV (0..180, по кругу) со знаком."""
    d = (h - ref + 90.0) % 180.0 - 90.0
    return d


def paleness(hsv: np.ndarray, s_w: float, v_w: float, k: Optional[int] = None) -> np.ndarray:
    """Насколько пиксель «бумажнее» дерева: падение насыщенности (доля от S дерева) + прирост светлоты
    (доля от V дерева), сглажено по окрестности — пачка купюр выделяется как компактное пятно, а не по пикселю."""
    S = hsv[..., 1].astype(np.float32)
    V = hsv[..., 2].astype(np.float32)
    ds = np.clip((s_w - S) / max(s_w, 1.0), -1.0, 1.0)
    dv = np.clip((V - v_w) / max(PALE_V_REL * v_w, 1.0), -1.0, 2.0)
    k = k or _blur_k(hsv.shape[0])
    return cv2.blur(ds + dv, (k, k))


def _blur_k(frame_h: int) -> int:
    """Окно сглаживания ~0.6 % высоты кадра (5 пикс. при 848), нечётное."""
    return max(3, int(round(frame_h / 170)) | 1)


def pale_candidates(frame: np.ndarray, desk: np.ndarray, pmask: np.ndarray, desk_area: int,
                    k: Optional[int] = None, offset: tuple[int, int] = (0, 0)) -> tuple[list[dict], float, float]:
    """Бледные компактные пятна на столешнице в одном кадре (без людей). Возвращает (кандидаты, S_дерева, V_дерева).
    frame/desk/pmask могут быть вырезкой вокруг стола: offset — её левый верхний угол, k — окно сглаживания
    по полному кадру."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    wood = _wood(hsv) & desk & ~pmask
    if wood.sum() < 0.1 * desk_area:
        return [], 0.0, 0.0
    s_w = float(np.median(hsv[..., 1][wood]))
    v_w = float(np.median(hsv[..., 2][wood]))
    h_w = float(np.median(hsv[..., 0][wood]))
    score = paleness(hsv, s_w, v_w, k)
    pale = (score > PALE_SCORE) & desk & ~pmask
    m = pale.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    out = []
    for i in range(1, n):
        x, y, w, h, a = (int(v) for v in st[i])
        rel = a / desk_area
        if rel < OBJ_MIN_REL or rel > OBJ_MAX_REL:
            continue
        fill = a / float(w * h)
        asp = max(w, h) / max(1.0, min(w, h))
        if fill < OBJ_MIN_FILL or asp > OBJ_MAX_ASPECT:
            continue
        sel = lab[y:y + h, x:x + w] == i
        hs = hsv[y:y + h, x:x + w][sel]
        out.append({
            "bbox": [x + offset[0], y + offset[1], x + w + offset[0], y + h + offset[1]], "area": a, "area_rel": round(rel, 5), "fill": round(fill, 3),
            "s_rel": round(float(hs[:, 1].mean()) / max(1.0, s_w), 3),
            "v_rel": round(float(hs[:, 2].mean()) - v_w, 1),
            "hue": round(float(np.median(hs[:, 0])), 1),
            "dhue": round(_hue_diff(float(np.median(hs[:, 0])), h_w), 1),
            "peak": round(float(score[y:y + h, x:x + w][sel].max()), 2),
            "white": round(float(((hs[:, 1] < 25) & (hs[:, 2] > 200)).mean()), 3),
        })
    return out, s_w, v_w


def compute_cash_track(path: str, info: VideoInfo, perc: Perception, progress=None,
                       samples: Optional[list[tuple[float, np.ndarray]]] = None) -> dict:
    """Проход по видео на CASH_FPS: бледные предметы на столешнице и их устойчивость во времени."""
    if samples is None:
        samples = sample_with_times(path, info, 25)
    gains = wb_gains(samples)
    desk = desk_support(samples, perc, gains)
    if desk is None:
        return {"version": CASH_VERSION, "fps": CASH_FPS, "desk": None, "wb_gains": [float(v) for v in gains], "frames": []}
    desk_area = int(desk.sum())
    ys, xs = np.nonzero(desk)
    dbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    step = max(1, int(round(info.fps / CASH_FPS)))
    dt = step / max(info.fps, 1e-6)
    tracks: list[dict] = []     # {bbox, area, first_t, last_seen, cand}
    rows: list[dict] = []
    total = max(1, info.frame_count)
    next_id = 0
    pad = 8
    cx1, cy1 = max(0, dbox[0] - pad), max(0, dbox[1] - pad)
    cx2, cy2 = min(desk.shape[1], dbox[2] + pad), min(desk.shape[0], dbox[3] + pad)
    desk_c = desk[cy1:cy2, cx1:cx2]
    k = _blur_k(desk.shape[0])
    for idx, t, frame in iter_frames(path, every_n=step):
        pm = _person_mask(frame.shape[:2], _persons_at(perc, t))
        cands, s_w, v_w = pale_candidates(_balance(frame[cy1:cy2, cx1:cx2], gains), desk_c, pm[cy1:cy2, cx1:cx2], desk_area,
                                          k=k, offset=(cx1, cy1))
        used = set()
        for tr in tracks:
            best = None
            for k, c in enumerate(cands):
                if k in used or not _near(tr["bbox"], c["bbox"]):
                    continue
                if best is None or _iou(tr["bbox"], c["bbox"]) > _iou(tr["bbox"], cands[best]["bbox"]):
                    best = k
            if best is not None:
                used.add(best)
                c = cands[best]
                tr.update(bbox=c["bbox"], cand=c, last_seen=t, seen=True, occluded=False, miss=0.0)
            else:
                x1, y1, x2, y2 = tr["bbox"]
                cov = float(pm[y1:y2, x1:x2].mean()) if (y2 > y1 and x2 > x1) else 1.0
                tr["seen"] = False
                tr["occluded"] = bool(cov >= 0.3 or s_w == 0.0)
                if not tr["occluded"]:
                    tr["miss"] += dt      # место видно, а предмета нет
        for k, c in enumerate(cands):
            if k not in used:
                tracks.append({"id": next_id, "bbox": c["bbox"], "cand": c, "first_t": t, "last_seen": t,
                               "seen": True, "occluded": False, "miss": 0.0})
                next_id += 1
        tracks = [tr for tr in tracks if tr["miss"] < MISS_SEC and t - tr["last_seen"] <= HOLD_SEC]
        objs = []
        for tr in tracks:
            c = tr["cand"]
            objs.append({
                "id": tr["id"], "bbox": tr["bbox"], "area": c["area"], "area_rel": c["area_rel"],
                "age_sec": round(t - tr["first_t"], 2), "seen": tr["seen"], "occluded": tr["occluded"],
                "s_rel": c["s_rel"], "v_rel": c["v_rel"], "hue": c["hue"], "fill": c["fill"], "white": c["white"],
                "peak": c["peak"], "dhue": c["dhue"],
            })
        objs.sort(key=lambda o: -o["area"])
        rows.append({"t": round(t, 2), "wood_s": round(s_w, 1), "wood_v": round(v_w, 1), "objs": objs[:12]})
        if progress and idx % (step * 20) == 0:
            progress("cash", int(100 * idx / total))
    return {"version": CASH_VERSION, "fps": CASH_FPS, "desk": {"bbox": dbox, "area": desk_area},
            "wb_gains": [round(float(v), 3) for v in gains], "frames": rows}


def cash_cache_path(cache: Path, stem: str) -> Path:
    return cache / "cash" / f"{stem}.json"


def load_or_compute(path: str, info: VideoInfo, perc: Perception, progress=None) -> dict:
    """Кеш results/cash/<видео>.json читается только при заданном FEATURES_CACHE; иначе — расчёт по видео."""
    stem = Path(path).stem.replace("X", "Х")
    cache = os.environ.get("FEATURES_CACHE")
    if cache:
        p = cash_cache_path(Path(cache), stem)
        if p.exists():
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                if d.get("version") == CASH_VERSION:
                    return d
            except Exception as exc:  # noqa: BLE001
                log.warning("cash cache %s unreadable: %s", p, exc)
    d = compute_cash_track(path, info, perc, progress=progress)
    if cache:
        p = cash_cache_path(Path(cache), stem)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    return d


# ---------------------------------------------------------------------------
# наличные на сетке времени сцены
# ---------------------------------------------------------------------------

def is_cashlike(o: dict, wood_v: float) -> bool:
    """Наблюдение бледного предмета похоже на пачку бежевых купюр (а не на документ, блик или кожу)."""
    return (abs(o.get("dhue", 99.0)) <= CASH_DHUE
            and CASH_S_REL[0] <= o["s_rel"] <= CASH_S_REL[1]
            and o["v_rel"] >= CASH_V_REL * max(wood_v, 1.0)
            and o["white"] < CASH_WHITE_MAX
            and o.get("peak", 0.0) >= CASH_PEAK)


def cash_tracks(cash: dict) -> set[int]:
    """Треки «пачка купюр»: большинство наблюдений похожи на пачку и предмет пролежал на месте ≥ CASH_STATIC_SEC
    (видимым или перекрытым человеком). Расчёт офлайн по всему ролику, поэтому предмет считается лежащим с
    момента, когда его положили, а не через CASH_STATIC_SEC после этого."""
    votes: dict[int, list[int]] = {}
    span: dict[int, list[float]] = {}
    for f in cash.get("frames", []):
        for o in f["objs"]:
            if o["seen"] or o["occluded"]:
                sp = span.setdefault(o["id"], [f["t"], f["t"]])
                sp[1] = f["t"]
            if o["seen"]:
                votes.setdefault(o["id"], []).append(int(is_cashlike(o, f.get("wood_v", 0.0))))
    return {k for k, v in votes.items()
            if len(v) >= CASH_TRACK_SEEN and np.mean(v) >= CASH_TRACK_VOTE and span[k][1] - span[k][0] >= CASH_STATIC_SEC}


def cash_on_grid(cash: Optional[dict], t_grid: np.ndarray) -> tuple[np.ndarray, list[Optional[list[int]]], np.ndarray]:
    """(present, bbox, area_rel) на сетке t_grid: лежит ли на столе пачка купюр (перекрытая человеком —
    продолжает «лежать»)."""
    n = len(t_grid)
    present = np.zeros(n, bool)
    boxes: list[Optional[list[int]]] = [None] * n
    area = np.zeros(n, float)
    if not cash or not cash.get("frames"):
        return present, boxes, area
    ids = cash_tracks(cash)
    if not ids:
        return present, boxes, area
    ct = np.asarray([f["t"] for f in cash["frames"]], float)
    for i, t in enumerate(t_grid):
        j = int(np.argmin(np.abs(ct - t)))
        if abs(ct[j] - t) > 0.6:
            continue
        objs = [o for o in cash["frames"][j]["objs"] if o["id"] in ids and (o["seen"] or o["occluded"])]
        if objs:
            best = max(objs, key=lambda o: o["area"])
            present[i] = True
            boxes[i] = list(best["bbox"])
            area[i] = float(sum(o["area_rel"] for o in objs))
    return present, boxes, area
