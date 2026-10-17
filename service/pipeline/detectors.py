"""Правила-детекторы T1–T13 (таксономия из ПЛАН_ForteBank.md, раздел 3).

Контракт: detect_tN(ctx: VideoContext) -> list[Event]. Таймкод события = момент перехода состояния
(кассир переступил границу зоны, свет погас, посторонний вошёл), а не середина эпизода (ПЛАН 2, 5.4).
Описание — только наблюдаемое, без «похитил/украл» (ПЛАН 4.4). Реализован T9; остальные — заглушки
с описанием признака и предполагаемыми входами из ctx.states/ctx.tracks, которые заполнит CV-слой.
"""
from __future__ import annotations

import numpy as np

from .context import Event, VideoContext

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
    return []


def detect_t2_safe_left_open(ctx: VideoContext) -> list[Event]:
    """T2. Сейф (хранилище) оставлен открытым.

    Признак: классификатор дверцы сейфа даёт 'open', при этом кассир не в зоне сейфа (отошёл к столу/вышел)
    дольше N с. Таймкод = момент, когда кассир покинул зону сейфа при открытой дверце.
    Входы: ctx.states['safe_open'], ctx.states['cashier_near_safe'], ctx.states['cashier_in_room'].
    Отрицательные примеры (сейф закрывают): X-21, X-19, X-39, X-5. Положительные: X-22, X-20.
    """
    return []


def detect_t3_workstation_unlocked(ctx: VideoContext) -> list[Event]:
    """T3. Незаблокированное рабочее место (АРМ) при уходе.

    Признак: кассир вышел из помещения (трек пересёк зону двери), экран ноутбука классифицирован как
    'active' (яркий, с содержимым), а не 'locked/dark'. Таймкод = момент выхода из помещения.
    Входы: ctx.states['cashier_in_room'], ctx.states['screen_unlocked']. Пример: X-28; отрицательный X-29.
    """
    return []


def detect_t4_unauthorized_person(ctx: VideoContext) -> list[Event]:
    """T4. Посторонний в кассовом помещении.

    Признак: число людей в помещении ≥ 3 или человек, не являющийся ни кассиром, ни клиентом на стуле клиента,
    находится в помещении дольше 1–2 с. Таймкод = момент, когда посторонний переступил порог (вошёл в зону комнаты).
    Входы: ctx.tracks (роли: cashier/client/other), ctx.zones['room'], ctx.zones['client_seat']. Примеры: X-18, X-33.
    Без распознавания лиц: роли только по положению и трекам.
    """
    return []


def detect_t5_client_in_work_zone(ctx: VideoContext) -> list[Event]:
    """T5. Клиент в рабочей зоне кассира.

    Признак: трек с ролью client/other пересёк линию стола и находится в рабочей зоне (сторона стола с ноутбуком,
    машинкой, сейфом). Таймкод = момент пересечения. Входы: ctx.tracks, ctx.zones['work_zone']. Пример: X-30.
    """
    return []


def detect_t6_count_without_client(ctx: VideoContext) -> list[Event]:
    """T6. Пересчёт/операция не в присутствии клиента.

    Признак: клиент покинул зону обслуживания (ушёл к двери/вышел), а кассир продолжает работать с наличными
    (деньги в руках/в машинке) ещё ≥ 2 с. Таймкод = момент выхода клиента из зоны обслуживания.
    Входы: ctx.states['client_present'], ctx.states['cash_handling']. Пример: X-34.
    """
    return []


def detect_t7_cash_exposed_on_entry(ctx: VideoContext) -> list[Event]:
    """T7. Наличные не убраны при входе клиента/постороннего.

    Признак: деньги на столе (cash_on_desk) в момент, когда новый трек входит в помещение.
    Таймкод = момент входа. Входы: ctx.states['cash_on_desk'], ctx.tracks (появление трека в зоне двери). Примеры: X-4, X-38.
    """
    return []


def detect_t8_personal_items(ctx: VideoContext) -> list[Event]:
    """T8. Личные вещи, еда, напитки на рабочем месте.

    Признак: детекция личной сумки / стакана / банки / телефона на столе кассира ≥ 2 с (устойчиво по трекy объекта),
    либо действие «пьёт» (стакан у лица). Таймкод = момент появления предмета на столе / начала действия.
    Входы: ctx.detections (классы handbag, cup, can, phone), ctx.zones['desk']. Примеры: X-40, X-46, X-24.
    """
    return []


def detect_t10_cash_handover_inside(ctx: VideoContext) -> list[Event]:
    """T10. Приём/передача ценностей внутри кассы посторонним лицом (инкассатор внутри кассы).

    Признак: трек с ролью other (инкассатор) внутри помещения у стола, детекция инкассаторской сумки переходит
    от него на стол/кассиру. Таймкод = момент входа инкассатора в помещение. Входы: ctx.tracks, ctx.detections['bag'].
    Пример: X-17; отрицательный (приём через передаточное устройство): X-14.
    """
    return []


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
    return []


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
