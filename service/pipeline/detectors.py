"""Правила-детекторы T1–T13 (таксономия из ПЛАН_ForteBank.md, раздел 3).

Контракт: detect_tN(ctx: VideoContext) -> list[Event]. Таймкод события = момент перехода состояния
(кассир переступил границу зоны, свет погас, посторонний вошёл), а не середина эпизода (ПЛАН 2, 5.4).
Описание — только наблюдаемое, без «похитил/украл» (ПЛАН 4.4). Реализованы T9 (контекст: свет выключен)
и T3 (экран АРМ остаётся активным в темноте после выключения света); остальные — заглушки с описанием
признака и предполагаемыми входами из ctx.states/ctx.tracks, которые заполнит CV-слой.
"""
from __future__ import annotations

from typing import Iterator, Optional

import cv2
import numpy as np

from . import rules_scene as _scene_rules
from .context import Event, VideoContext
from .video import VideoInfo, open_capture

# ---------------------------------------------------------------------------
# T9 — реализовано: потеря видеоконтроля (освещение выключено)
# ---------------------------------------------------------------------------

T9_DROP_RATIO = 0.60        # кадр считается «тёмным», если яркость < 60 % опорной
T9_HOLD_RATIO = 0.80        # после падения яркость должна держаться < 80 % опорной…
T9_HOLD_SEC = 1.0           # …не менее 1 с (ПЛАН: «падение < порога на > 1 с»)
T9_HOLD_FRACTION = 0.8      # доля тёмных отсчётов в окне удержания
T9_REF_WINDOW = (3.0, 0.3)  # опорная яркость = медиана за [t-3.0; t-0.3] с
T9_MIN_REF = 40.0           # если и до события было темно (< 40/255) — не судим
T9_RECOVER_RATIO = 0.9      # свет «включён», когда яркость ≥ 90 % опорной 1 с подряд


def detect_t9_lights_off(ctx: VideoContext) -> list[Event]:
    """T9. Резкое падение средней яркости кадра относительно предыдущих секунд, удерживающееся > 1 с.

    Камера с автоэкспозицией частично компенсирует темноту (X-11: 138 → 19 → ~90), поэтому опорное
    значение берётся из окна ДО падения, а не скользящим образом после него.
    """
    t, v = ctx.bright_t, ctx.bright_v
    n = len(t)
    if n < 5:
        return []
    events: list[Event] = []
    i = 0
    while i < n:
        lo, hi = t[i] - T9_REF_WINDOW[0], t[i] - T9_REF_WINDOW[1]
        ref_mask = (t >= lo) & (t <= hi)
        if ref_mask.sum() < 3:
            i += 1
            continue
        ref = float(np.median(v[ref_mask]))
        if ref < T9_MIN_REF or v[i] >= T9_DROP_RATIO * ref:
            i += 1
            continue
        # кандидат на событие: проверяем удержание
        hold_mask = (t >= t[i]) & (t <= t[i] + T9_HOLD_SEC)
        hold_vals = v[hold_mask]
        if len(hold_vals) < 3 or (hold_vals < T9_HOLD_RATIO * ref).mean() < T9_HOLD_FRACTION:
            i += 1
            continue
        start = float(t[i])
        min_v = float(hold_vals.min())
        # ищем момент восстановления освещения
        end = float(ctx.info.duration_sec)
        recovered = False
        j = i
        while j < n:
            win = (t >= t[j]) & (t <= t[j] + 1.0)
            if win.sum() >= 3 and (v[win] >= T9_RECOVER_RATIO * ref).all():
                end = float(t[j])
                recovered = True
                break
            j += 1
        dark_dur = end - start
        drop_pct = 100.0 * (1.0 - min_v / ref)
        conf = min(0.99, 0.6 + 0.3 * min(1.0, drop_pct / 80.0) + 0.1 * min(1.0, dark_dur / 5.0))
        tail = f"и свет включился в {end:.1f} с" if recovered else f"и оставалась пониженной до конца ролика ({end:.1f} с)"
        desc = (
            f"В {start:.1f} с средняя яркость кадра упала с {ref:.0f} до {min_v:.0f} (на {drop_pct:.0f} %) "
            f"{tail}. Освещение в кассовом помещении выключено, визуальный контроль за кассой утрачен: "
            f"действия с наличными и доступ к хранилищу в этот период не наблюдаемы по видеозаписи."
        )
        after = min(start + 2.0, max(start + 0.5, end - 0.2)) if recovered else min(start + 2.0, ctx.info.duration_sec - 0.1)
        before = max(0.0, start - 1.0)
        ev = Event(
            type_code="T9",
            timecode_sec=round(start, 2),
            end_sec=round(end, 2),
            confidence=round(conf, 2),
            description=desc,
            evidence_times=[before, start, after],
            evidence_captions=[
                f"до события: яркость {ref:.0f}",
                f"освещение выключено: яркость {min_v:.0f} (−{drop_pct:.0f} %)",
                "через 2 с: помещение без освещения",
            ],
            extras={"ref": ref, "min": min_v, "dark_sec": dark_dur, "recovered": recovered},
        )
        events.append(ev)
        # пропускаем тёмный участок
        i = max(j, i + 1) if recovered else n
    return events


# ---------------------------------------------------------------------------
# Заглушки T1–T8, T10–T13: контракт и ожидаемые входы. Заполняет CV-команда.
# ---------------------------------------------------------------------------

def detect_t1_valuables_unattended(ctx: VideoContext) -> list[Event]:
    """T1. Наличные/ценности оставлены без присмотра.

    Признак: кассир вне зоны стола дольше N с (N ≈ 3–5, относительный к длительности), при этом на столе
    детектированы деньги (пачка/купюры) или инкассаторская сумка. Таймкод = момент выхода кассира из зоны стола.
    Входы: ctx.states['cashier_at_desk'], ctx.states['cash_on_desk'], ctx.states['bag_on_desk'].
    Кадры: момент выхода (с рамкой вокруг ценностей на столе), +1 с, +3 с. Примеры: X-15, X-26.
    """
    return _scene_rules.rule_t1(ctx)


def detect_t2_safe_left_open(ctx: VideoContext) -> list[Event]:
    """T2. Сейф (хранилище) оставлен открытым.

    Признак: классификатор дверцы сейфа даёт 'open', при этом кассир не в зоне сейфа (отошёл к столу/вышел)
    дольше N с. Таймкод = момент, когда кассир покинул зону сейфа при открытой дверце.
    Входы: ctx.states['safe_open'], ctx.states['cashier_near_safe'], ctx.states['cashier_in_room'].
    Отрицательные примеры (сейф закрывают): X-21, X-19, X-39, X-5. Положительные: X-22, X-20.
    """
    return _scene_rules.rule_t2(ctx)


# ---------------------------------------------------------------------------
# T3 — реализовано: экран АРМ остаётся активным после выключения света (уход в темноте)
# ---------------------------------------------------------------------------

T3_SAMPLE_FPS = 10.0          # частота выборки кадров тёмного участка
T3_MIN_DARK_SEC = 2.0         # тёмный участок короче 2 с не анализируем (стабильность не проверить)
T3_BRIGHT = 150               # пиксель «светится», если серый > 150 (экран в темноте: 175 без усиления, 230+ с усилением)
T3_STABLE_FRACTION = 0.8      # область стабильна, если пиксели светятся не менее чем в 80 % тёмных кадров
T3_AREA_RANGE = (0.0003, 0.01)  # площадь области, доля кадра: экран ноутбука ~0.14 %; окно/дверь в освещённый
                                # коридор — 1.5–2.5 % (исключаются по площади и по контрасту)
T3_MIN_FILL = 0.4             # заполненность bbox: экран — компактный прямоугольник (0.5), блики/кромки — < 0.4
T3_MIN_MEAN = 170             # средняя яркость области в тёмных кадрах
T3_MIN_CONTRAST = 3.0         # яркость области / яркость кольца вокруг неё: самосветящийся объект в тёмной комнате
                              # (экран: ×4.6–5.1); проёмы в освещённый коридор, флипчарт за стеклом: ≤ ×2.6
T3_RING = 0.5                 # ширина кольца вокруг bbox — 50 % размера bbox (+4 пикс.)


def _iter_segment(path: str, info: VideoInfo, t0: float, t1: float, every_n: int) -> Iterator[tuple[float, np.ndarray]]:
    """Кадры (время, BGR) участка [t0; t1] с шагом every_n кадров.

    Сначала пробуем позиционироваться по индексу кадра; если бэкенд сообщает другую позицию или не отдаёт
    кадр (так ведёт себя Х-11.mov с fps 29.88), читаем последовательно с начала, пропуская кадры через grab().
    """
    start = max(0, int(round(t0 * info.fps)))
    stop = min(max(info.frame_count - 1, start), int(round(t1 * info.fps)))
    cap = open_capture(path)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        seek_ok = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) == start
        if not seek_ok:
            cap.release()
            cap = open_capture(path)
            for _ in range(start):
                if not cap.grab():
                    return
        idx = start
        while idx <= stop:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if (idx - start) % every_n == 0:
                yield idx / info.fps, frame
            idx += 1
    finally:
        cap.release()


def _ring_mean(gray: np.ndarray, x: int, y: int, w: int, h: int) -> float:
    """Средняя яркость кольца вокруг bbox (bbox, расширенный на T3_RING с каждой стороны, минус сам bbox)."""
    H, W = gray.shape
    px, py = int(w * T3_RING) + 4, int(h * T3_RING) + 4
    x0, y0, x1, y1 = max(0, x - px), max(0, y - py), min(W, x + w + px), min(H, y + h + py)
    ring = np.ones((y1 - y0, x1 - x0), dtype=bool)
    ring[y - y0:y - y0 + h, x - x0:x - x0 + w] = False
    return float(gray[y0:y1, x0:x1][ring].mean()) if ring.any() else 255.0


def find_glowing_regions(path: str, info: VideoInfo, t0: float, t1: float) -> list[dict]:
    """Стабильно светящиеся компактные области на тёмном участке [t0; t1] (кандидаты «экран активен»).

    Карта «доля тёмных кадров, в которых пиксель светится» → компоненты связности с долей ≥ 80 % →
    фильтр по площади, заполненности bbox, яркости и локальному контрасту (область / кольцо вокруг неё)
    на усреднённом тёмном кадре. Движущиеся объекты (человек в светлой одежде) размазываются по карте и
    не образуют стабильной области; проёмы в освещённый коридор стабильны, но крупные и низкоконтрастные.
    """
    every_n = max(1, int(round(info.fps / T3_SAMPLE_FPS)))
    n = 0
    bright_cnt: Optional[np.ndarray] = None
    gray_sum: Optional[np.ndarray] = None
    bgr_sum: Optional[np.ndarray] = None
    for _t, frame in _iter_segment(path, info, t0, t1, every_n):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if bright_cnt is None:
            bright_cnt = np.zeros(gray.shape, dtype=np.float32)
            gray_sum = np.zeros(gray.shape, dtype=np.float32)
            bgr_sum = np.zeros(frame.shape, dtype=np.float32)
        bright_cnt += gray > T3_BRIGHT
        gray_sum += gray
        bgr_sum += frame
        n += 1
    if n < 3 or bright_cnt is None:
        return []
    stable = (bright_cnt / n >= T3_STABLE_FRACTION).astype(np.uint8)
    mean_gray = gray_sum / n
    mean_bgr = bgr_sum / n
    H, W = stable.shape
    num, labels, stats, _ = cv2.connectedComponentsWithStats(stable, connectivity=8)
    out: list[dict] = []
    for i in range(1, num):
        x, y, w, h, area = (int(v) for v in stats[i])
        area_frac = area / float(W * H)
        if not (T3_AREA_RANGE[0] <= area_frac <= T3_AREA_RANGE[1]):
            continue
        fill = area / float(w * h)
        sel = labels == i
        mean = float(mean_gray[sel].mean())
        ring = _ring_mean(mean_gray, x, y, w, h)
        contrast = mean / max(ring, 1.0)
        b, _g, r = (float(mean_bgr[..., k][sel].mean()) for k in range(3))
        stability = float((bright_cnt[sel] / n).mean())
        cand = {
            "xyxy": (x, y, x + w, y + h), "area": area, "area_frac": area_frac, "fill": fill, "mean": mean,
            "ring": ring, "contrast": contrast, "blue_minus_red": b - r, "stability": stability, "frames": n,
        }
        if fill >= T3_MIN_FILL and mean >= T3_MIN_MEAN and contrast >= T3_MIN_CONTRAST:
            out.append(cand)
    out.sort(key=lambda c: -c["contrast"] * c["area"])
    return out


def detect_t3_workstation_unlocked(ctx: VideoContext) -> list[Event]:
    """T3. Незаблокированное рабочее место (АРМ) при уходе.

    Два варианта: (а) при освещении — экран активен и кассир ушёл от стола (rules_scene.rule_t3_lit, X-28);
    (б) в темноте — после выключения света экран продолжает светиться (ниже, X-11).

    Реализованный признак (по контексту T9): после выключения освещения в кассе (кассир гасит свет, уходя)
    в тёмных кадрах сохраняется компактная, яркая, неподвижная область с высоким локальным контрастом —
    экран ноутбука остаётся активным (Х-11). В чистом Х-12 свет тоже гаснет, но крышка ноутбука закрыта
    и такой области нет. Таймкод = момент выключения света (момент ухода), end_sec = конец тёмного участка.
    Без контекста T9 правило молчит. Будущий вход CV-слоя: ctx.states['cashier_in_room'] и
    ctx.states['screen_unlocked'] (уход при включённом свете — X-28; отрицательный X-29).
    """
    events: list[Event] = []
    for t9 in [e for e in ctx.context_events if e.type_code == "T9"]:
        t0 = float(t9.timecode_sec)
        t1 = float(t9.end_sec if t9.end_sec is not None else ctx.info.duration_sec)
        if t1 - t0 < T3_MIN_DARK_SEC:
            continue
        regions = find_glowing_regions(ctx.path, ctx.info, t0, t1)
        if not regions:
            continue
        reg = regions[0]
        x1, y1, x2, y2 = reg["xyxy"]
        dark_dur = t1 - t0
        conf = 0.7 + 0.15 * min(1.0, (reg["contrast"] - T3_MIN_CONTRAST) / 2.0) + 0.1 * min(1.0, dark_dur / 5.0)
        if reg["blue_minus_red"] >= 15:   # холодный (голубоватый) оттенок свечения — характерно для экрана
            conf += 0.04
        conf = min(0.97, conf)
        recovered = bool(t9.extras.get("recovered"))
        till = f"до включения света в {t1:.1f} с" if recovered else f"до конца записи ({t1:.1f} с)"
        desc = (
            f"В {t0:.1f} с освещение в кассе выключено (свет гасят, покидая рабочее место), при этом экран "
            f"ноутбука (АРМ) остаётся активным: в темноте {till} на месте экрана сохраняется неподвижная "
            f"светящаяся область {x2 - x1}×{y2 - y1} пикс. (яркость {reg['mean']:.0f} при {reg['ring']:.0f} вокруг, "
            f"контраст ×{reg['contrast']:.1f}). Рабочее место при уходе не заблокировано — экран не погашен."
        )
        # кадры: до выключения света (экран включён), самый тёмный кадр сразу после выключения (экран светится
        # в темноте), конец тёмного участка (экран по-прежнему активен)
        bt, bv = ctx.bright_t, ctx.bright_v
        win = (bt >= t0) & (bt <= t0 + 0.5)
        darkest = float(bt[win][np.argmin(bv[win])]) if win.any() else t0
        before = max(0.0, t0 - 1.0)
        last = max(darkest + 0.5, t1 - 0.3)
        ev = Event(
            type_code="T3",
            timecode_sec=round(t0, 2),
            end_sec=round(t1, 2),
            confidence=round(conf, 2),
            description=desc,
            evidence_times=[before, darkest, last],
            evidence_captions=[
                "до выключения света: экран ноутбука включён",
                "освещение выключено: экран светится в темноте",
                "экран по-прежнему активен, рабочее место не заблокировано",
            ],
            extras={"region": reg, "evidence_boxes": [reg["xyxy"]] * 3, "t9": t9.timecode_sec},
        )
        events.append(ev)
    if not events:
        events.extend(_scene_rules.rule_t3_lit(ctx))
    return events


def detect_t4_unauthorized_person(ctx: VideoContext) -> list[Event]:
    """T4. Посторонний в кассовом помещении.

    Признак: число людей в помещении ≥ 3 или человек, не являющийся ни кассиром, ни клиентом на стуле клиента,
    находится в помещении дольше 1–2 с. Таймкод = момент, когда посторонний переступил порог (вошёл в зону комнаты).
    Входы: ctx.tracks (роли: cashier/client/other), ctx.zones['room'], ctx.zones['client_seat']. Примеры: X-18, X-33.
    Без распознавания лиц: роли только по положению и трекам.
    """
    return _scene_rules.rule_t4(ctx)


def detect_t5_client_in_work_zone(ctx: VideoContext) -> list[Event]:
    """T5. Клиент в рабочей зоне кассира.

    Признак: трек с ролью client/other пересёк линию стола и находится в рабочей зоне (сторона стола с ноутбуком,
    машинкой, сейфом). Таймкод = момент пересечения. Входы: ctx.tracks, ctx.zones['work_zone']. Пример: X-30.
    """
    return _scene_rules.rule_t5(ctx)


def detect_t6_count_without_client(ctx: VideoContext) -> list[Event]:
    """T6. Пересчёт/операция не в присутствии клиента.

    Признак: клиент покинул зону обслуживания (ушёл к двери/вышел), а кассир продолжает работать с наличными
    (деньги в руках/в машинке) ещё ≥ 2 с. Таймкод = момент выхода клиента из зоны обслуживания.
    Входы: ctx.states['client_present'], ctx.states['cash_handling']. Пример: X-34.
    """
    return _scene_rules.rule_t6(ctx)


def detect_t7_cash_exposed_on_entry(ctx: VideoContext) -> list[Event]:
    """T7. Наличные не убраны при входе клиента/постороннего.

    Признак: деньги на столе (cash_on_desk) в момент, когда новый трек входит в помещение.
    Таймкод = момент входа. Входы: ctx.states['cash_on_desk'], ctx.tracks (появление трека в зоне двери). Примеры: X-4, X-38.
    """
    return _scene_rules.rule_t7(ctx)


def detect_t8_personal_items(ctx: VideoContext) -> list[Event]:
    """T8. Личные вещи, еда, напитки на рабочем месте.

    Признак: детекция личной сумки / стакана / банки / телефона на столе кассира ≥ 2 с (устойчиво по трекy объекта),
    либо действие «пьёт» (стакан у лица). Таймкод = момент появления предмета на столе / начала действия.
    Входы: ctx.detections (классы handbag, cup, can, phone), ctx.zones['desk']. Примеры: X-40, X-46, X-24.
    """
    return _scene_rules.rule_t8(ctx)


def detect_t10_cash_handover_inside(ctx: VideoContext) -> list[Event]:
    """T10. Приём/передача ценностей внутри кассы посторонним лицом (инкассатор внутри кассы).

    Признак: трек с ролью other (инкассатор) внутри помещения у стола, детекция инкассаторской сумки переходит
    от него на стол/кассиру. Таймкод = момент входа инкассатора в помещение. Входы: ctx.tracks, ctx.detections['bag'].
    Пример: X-17; отрицательный (приём через передаточное устройство): X-14.
    """
    return _scene_rules.rule_t10(ctx)


def detect_t11_cash_moved_out_of_zone(ctx: VideoContext) -> list[Event]:
    """T11. Признак перемещения наличных вне кассовой зоны (в карман/к коленям/в сумку).

    Признак: по позе кассира — купюры в руке, затем кисть уходит ниже уровня стола/к карману и купюры исчезают
    из кадра без появления на столе/в машинке/в сейфе. Индикатор с повышенным риском ложных срабатываний:
    выдавать с пониженной уверенностью (≤ 0.6) и пометкой «требует проверки». Таймкод = момент ухода руки вниз.
    Входы: ctx.cache['pose'], ctx.detections['cash']. Примеры (неуверенные): X-2, X-44, X-1, X-13.
    """
    return []


def detect_t12_no_machine_count(ctx: VideoContext) -> list[Event]:
    """T12. Приём/выдача без пересчёта и проверки на оборудовании.

    Признак: наличные приняты от клиента / выданы клиенту, но за время операции детекции денег не попадали в зону
    счётной машинки/детектора. Индикатор с повышенным риском ложных срабатываний; пониженная уверенность.
    Таймкод = момент передачи денег. Входы: ctx.detections['cash'], ctx.zones['counter_machine']. Пример: X-47.
    """
    return []


def detect_t13_empty_cash_desk_door_open(ctx: VideoContext) -> list[Event]:
    """T13. Касса без кассира с открытой дверью.

    Признак: в помещении 0 людей ≥ 2 с и дверь классифицирована как открытая. Таймкод = момент, когда условие
    стало истинным (последний человек вышел при открытой двери / дверь открылась в пустой кассе).
    Входы: ctx.states['n_people'], ctx.states['door_open']. Пример: X-32.
    """
    return _scene_rules.rule_t13(ctx)


# Порядок = порядок кодов. Выключить правило — убрать из списка.
DETECTORS = [
    detect_t1_valuables_unattended,
    detect_t2_safe_left_open,
    detect_t3_workstation_unlocked,
    detect_t4_unauthorized_person,
    detect_t5_client_in_work_zone,
    detect_t6_count_without_client,
    detect_t7_cash_exposed_on_entry,
    detect_t8_personal_items,
    detect_t9_lights_off,
    detect_t10_cash_handover_inside,
    detect_t11_cash_moved_out_of_zone,
    detect_t12_no_machine_count,
    detect_t13_empty_cash_desk_door_open,
]
