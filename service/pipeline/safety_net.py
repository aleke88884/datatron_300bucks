"""Страховочная сетка (safety net) для типов нарушений, которые правила T1–T13 не поймали (FAQ п. 9).

Идея — «детектор предлагает, VLM подтверждает, контроль опровергает». Модель НИКОГДА не ищет нарушения сама по
всему видео (так она галлюцинирует) и НИКОГДА не придумывает тип или пункт закона: она только отвечает «да/нет»
на узкий вопрос по увеличенному фрагменту кадра вокруг предмета, который уже несколько раз увидел детектор.

Два слоя:

1. Структурный (без VLM, работает всегда). Признак, которого нет в правилах T1–T13 и который физически
   однозначен: посетитель (не кассир) остаётся в кассе один после ухода кассира (S1). Норма — из RULES["T4"]
   (Правила № 14 п. 8, п. 16 пп. 1): изоляция кассового узла); п. 75 Правил № 231 касается входа в кладовую/
   сейфовую комнату, а не в кассу, и здесь НЕ используется. Смежно — п. 22 Правил № 231 (кассовому работнику не
   допускается покидать рабочее место до окончания обслуживания клиента). В 42 обучающих роликах не встречается ни разу (ни в чистых, ни в нарушениях).

2. Предметный (VLM, только если модель доступна и успевает). Предложения — повторяющиеся детекции COCO-предметов
   YOLOX у рабочего места кассира, которые правила не используют или отбрасывают как шумные (телефон, напиток/еда,
   личная сумка). Событие добавляется только при ВСЕХ условиях:
     * детектор видел предмет этой группы ≥ PROP_MIN_DET раз за ≥ PROP_MIN_SPAN_SEC в одном эпизоде, предмет не является статичным
       интерьером (виден в большей части ролика) и связан с кассиром (в рамке кассира или в зоне стола, не у
       посетителя);
     * VLM ответила «да» на узкий вопрос по увеличенному фрагменту минимум на VERIFY_MIN_YES кадрах, разнесённых
       не меньше чем на VERIFY_MIN_GAP_SEC;
     * VLM ответила «нет» на вопросы-опровержения (например, «это купюры, а не телефон?») на тех же кадрах;
     * VLM ответила «нет» на тот же вопрос по ТОМУ ЖЕ фрагменту кадра в контрольный момент, когда детектор этого
       предмета там не видел (защита от «модель всегда говорит да на этот участок стола»);
     * «не знаю»/неразборчивый ответ считается «нет».
   Тип и норма берутся из RULES (сейчас все предметные пробы → T8, Правила № 231 п. 12: хранение личных вещей
   работников — вне кассового узла; еда и напитки в пункте прямо не названы, средства связи без фото-/видеофункции
   вносить допускается — это оговаривается в описании события). Описание — фиксированный русский шаблон с наблюдаемыми фактами, без свободного
   текста модели.

Политика включения (переменные окружения):
    SAFETY_NET            auto (по умолчанию) | 1 | 0. auto — предметный слой работает, только если VLM включена,
                          загружена и прошла замер скорости; 1 — то же, плюс разрешены ответы из кеша при
                          FEATURES_CACHE (офлайн-оценка без модели); 0 — сетка выключена целиком.
    SAFETY_NET_STRUCTURAL 1 (по умолчанию) | 0 — структурный слой S1.
    SAFETY_NET_ALWAYS     0 (по умолчанию) — предметный слой запускается только если правила ничего не нашли
                          (именно там пропуск «недопустим», FAQ 16); 1 — всегда.
    SAFETY_NET_BUDGET_SEC бюджет времени предметного слоя на видео (по умолчанию 45 с).

Кеш ответов (только при FEATURES_CACHE): results/safety_net/<видео>.json — ключ вопроса → true/false/null.
Только ответы модели, никаких кадров и текстов.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

import numpy as np

from .context import Event, VideoContext

log = logging.getLogger("pipeline.safety_net")

SOURCE = "safety_net"

# --- структурный слой ----------------------------------------------------
S1_ALONE_SEC = 3.0            # посетитель один в кассе не менее 3 с
S1_CASHIER_BEFORE_SEC = 1.0   # кассир и посетитель были внутри одновременно не менее 1 с до ухода кассира

# --- предметный слой -----------------------------------------------------
PROP_MIN_DET = 3              # детекций предмета в одном эпизоде (3 к/с: предмет реально виден ≥ 1 с) …
PROP_MIN_SPAN_SEC = 1.0       # … и эпизод длится ≥ 1 с: мгновенные ошибки детектора (веер купюр как «бокал», Х-1) отсекаются
PROP_GAP_SEC = 2.0            # разрыв между детекциями внутри эпизода
PROP_STATIC_FRAC = 0.5        # предмет виден в той же точке в ≥ 50 % кадров ролика → интерьер, не событие
PROP_STATIC_IOU = 0.3
VERIFY_MIN_YES = 2            # «да» минимум на двух кадрах …
VERIFY_MIN_GAP_SEC = 1.0      # … разнесённых не меньше чем на 1 с
VERIFY_MAX_FRAMES = 3
CONTROL_CLEAR_SEC = 1.5       # контрольный момент: детектор не видел предмет этой группы в этой точке ±1.5 с
CROP_SCALE = 3.0              # фрагмент = рамка предмета × 3 (контекст: руки, стол)
CROP_MIN_FRAC = 0.30          # не меньше 30 % ширины кадра
MAX_PROPOSALS = 4             # не более 4 эпизодов на видео (бюджет времени)
FOLLOW_STEP_SEC = 1.5         # «куда делся внесённый предмет»: шаг проверки столешницы после эпизода …
FOLLOW_MAX_FRAMES = 8         # … не более 8 кадров (≈ 12 с после эпизода)
FOLLOW_START_SEC = 2.0
DEFAULT_BUDGET_SEC = 45.0

_ANSWER = (
    " Answer with one word: yes or no. If you are not sure, answer no."
)


@dataclass(frozen=True)
class Probe:
    """Предметная проба: какие классы детектора её предлагают, что спросить у VLM и куда отнести событие."""

    key: str
    classes: frozenset
    type_code: str
    question: str                         # «да» = предмет есть
    refute: tuple = ()                    # вопросы-опровержения: «да» на любом → отказ
    what_ru: str = ""                     # что видно — для описания
    caption_ru: str = ""
    follow_desk: bool = False             # предмет внесли — проверить, не оказался ли он потом на столе
    desk_question: str = ""
    hedge_ru: str = ""                    # оговорка о том, чего по видео не видно (для описания)


PROBES: dict[str, Probe] = {
    "phone": Probe(
        key="phone",
        classes=frozenset({"cell phone", "remote"}),
        type_code="T8",
        question="Look at the middle of the picture. Is a person holding or using a mobile phone (smartphone) there?",
        refute=("Is the person in the middle of the picture holding banknotes or paper money?",),
        what_ru="мобильный телефон в руках кассира",
        caption_ru="телефон у кассира на рабочем месте",
        hedge_ru=("Есть ли у телефона функция фото- и видеофиксации, по видео не определяется (п. 12 Правил № 231 "
                  "допускает внос средств связи без такой функции)."),
    ),
    "drink": Probe(
        key="drink",
        classes=frozenset({"cup", "bottle", "wine glass", "bowl"}),
        type_code="T8",
        question="Look at the middle of the picture. Is there a cup, mug, glass, can, bottle or food there?",
        refute=("Is the person in the middle of the picture holding banknotes or paper money?",),
        what_ru="напиток или еда (стакан/бутылка/посуда)",
        caption_ru="напиток/еда на рабочем месте кассира",
        hedge_ru="Еда и напитки в п. 12 Правил № 231 прямо не названы.",
    ),
    "bag": Probe(
        key="bag",
        classes=frozenset({"handbag", "backpack", "suitcase"}),
        type_code="T8",
        question="Look at the middle of the picture. Is there a personal handbag, purse or backpack lying on the desk there?",
        refute=("Is the bag in the middle of the picture a green cash-collection bag?",),
        what_ru="личная сумка на рабочем столе",
        caption_ru="личная сумка на рабочем столе кассира",
        follow_desk=True,
        desk_question="Is there a personal handbag, purse or backpack lying on top of the desk in this picture?",
    ),
}


class Verifier(Protocol):
    """Источник ответов «да/нет» по фрагменту кадра (BGR). None — нет ответа (считается «нет»)."""

    def ask_yes_no(self, image: np.ndarray, question: str) -> Optional[bool]: ...


@dataclass
class Proposal:
    probe: str
    times: list[float]
    boxes: list[tuple[float, float, float, float]]
    scores: list[float]

    @property
    def t0(self) -> float:
        return self.times[0]

    @property
    def t1(self) -> float:
        return self.times[-1]

    @property
    def n(self) -> int:
        return len(self.times)

    @property
    def best(self) -> int:
        return int(np.argmax(self.scores))

    @property
    def box(self) -> tuple[float, float, float, float]:
        return self.boxes[self.best]


@dataclass
class NetReport:
    """Что делала сетка на этом видео (уходит в meta.safety_net ответа)."""

    enabled: bool = False
    vlm: str = "off"                    # off | unavailable | slow | ready | cache
    reason: str = ""
    proposals: int = 0
    verified: int = 0
    vlm_calls: int = 0
    elapsed_sec: float = 0.0
    details: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "enabled": self.enabled, "vlm": self.vlm, "reason": self.reason, "proposals": self.proposals,
            "verified": self.verified, "vlm_calls": self.vlm_calls, "elapsed_sec": round(self.elapsed_sec, 2),
            "details": self.details[:8],
        }


# ---------------------------------------------------------------------------
# утилиты
# ---------------------------------------------------------------------------

def _env(name: str, default: str) -> str:
    return (os.environ.get(name) or default).strip().lower()


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _center(b) -> tuple[float, float]:
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def _inside(pt, box, pad: float = 0.0) -> bool:
    w, h = box[2] - box[0], box[3] - box[1]
    return box[0] - pad * w <= pt[0] <= box[2] + pad * w and box[1] - pad * h <= pt[1] <= box[3] + pad * h


def _work_area(scene) -> Optional[tuple[float, float, float, float]]:
    """Зона рабочего места: рамка стола, расширенная вверх (предметы в руках сидящего/стоящего кассира) —
    относительно размеров самого стола, без пиксельных констант камеры."""
    desk = (scene.anchors or {}).get("desk") if scene is not None else None
    if not desk:
        return None
    x1, y1, x2, y2 = desk["bbox"]
    w, h = x2 - x1, y2 - y1
    return (x1 - 0.15 * w, y1 - 0.8 * h, x2 + 0.15 * w, y2 + 0.1 * h)


def crop_box(box, width: int, height: int) -> tuple[int, int, int, int]:
    """Квадратный фрагмент вокруг рамки предмета: сторона = max(w, h) × CROP_SCALE, не меньше CROP_MIN_FRAC ширины."""
    cx, cy = _center(box)
    side = max(box[2] - box[0], box[3] - box[1]) * CROP_SCALE
    side = max(side, CROP_MIN_FRAC * width)
    side = min(side, float(min(width, height)))
    x1 = int(round(min(max(0.0, cx - side / 2), width - side)))
    y1 = int(round(min(max(0.0, cy - side / 2), height - side)))
    return (x1, y1, int(round(x1 + side)), int(round(y1 + side)))


def _crop(frame: np.ndarray, box) -> np.ndarray:
    h, w = frame.shape[:2]
    if len(box) == 5 and box[4] == "area":       # прямоугольник как есть (столешница целиком)
        x1, y1, x2, y2 = (int(max(0, box[0])), int(max(0, box[1])), int(min(w, box[2])), int(min(h, box[3])))
    else:
        x1, y1, x2, y2 = crop_box(box, w, h)
    return frame[y1:y2, x1:x2]


def _desk_area_box(scene) -> Optional[tuple]:
    """Столешница целиком (рамка стола + 10 % и немного вверх) — для вопроса «что лежит на столе»."""
    desk = (scene.anchors or {}).get("desk") if scene is not None else None
    if not desk:
        return None
    x1, y1, x2, y2 = desk["bbox"]
    w, h = x2 - x1, y2 - y1
    return (x1 - 0.1 * w, y1 - 0.35 * h, x2 + 0.1 * w, y2 + 0.1 * h, "area")


# ---------------------------------------------------------------------------
# структурный слой
# ---------------------------------------------------------------------------

def structural_events(scene, events: list[Event]) -> list[Event]:
    """S1: кассир ушёл из кассы, а посетитель, бывший внутри вместе с ним, остался в помещении один.

    Состояние в момент ухода, а не само действие: в чистых роликах кассир уходит только из пустой кассы.
    Тип — T4 (третье лицо в кассовом помещении без кассира), если правило T4 ещё не сработало.
    """
    if scene is None or len(scene.t) == 0:
        return []
    if any(e.type_code == "T4" for e in events):
        return []
    from .scene import GRID_FPS

    dt = 1.0 / GRID_FPS
    lit = ~scene.lights_off
    alone = (~scene.cashier_inside) & (scene.n_visitors >= 1) & lit
    for a, b in scene.runs(alone, S1_ALONE_SEC, fill_gap_sec=0.5):
        # кассир был внутри вместе с ЭТИМ ЖЕ посетителем непосредственно перед уходом
        vis_now = set().union(*[set(scene.visitor_ids[i]) for i in range(a, b + 1)]) if b >= a else set()
        lo = max(0, a - int(6 * GRID_FPS))
        together = [i for i in range(lo, a) if scene.cashier_inside[i] and vis_now & set(scene.visitor_ids[i])]
        if len(together) * dt < S1_CASHIER_BEFORE_SEC:
            continue
        t_leave = float(scene.t[together[-1]]) + dt
        dur = (b - a + 1) * dt
        desc = (
            f"В {t_leave:.1f} с кассир покидает кассовое помещение, а посетитель, находившийся внутри вместе с ним, "
            f"остаётся в кассе один не менее {dur:.1f} с (до {float(scene.t[b]):.1f} с). Может указывать на присутствие "
            f"третьего лица в кассовом узле без кассира; признак для проверки человеком: допущен ли этот человек в "
            f"кассовый узел и было ли завершено обслуживание к моменту ухода кассира (п. 22 Правил № 231)."
        )
        return [Event(
            type_code="T4", timecode_sec=round(t_leave, 2), end_sec=round(float(scene.t[b]), 2), confidence=0.7,
            description=desc,
            evidence_times=[max(0.0, t_leave - 0.5), float(scene.t[a]) + 0.5, float(scene.t[min(b, a + int(2 * GRID_FPS))])],
            evidence_captions=["кассир уходит, посетитель внутри", "посетитель в кассе без кассира", "посетитель остаётся один"],
            extras={"source": SOURCE, "probe": "S1_visitor_alone", "alone_sec": dur},
        )]
    return []


# ---------------------------------------------------------------------------
# предложения детектора
# ---------------------------------------------------------------------------

def item_proposals(scene, perc) -> list[Proposal]:
    """Эпизоды повторяющихся детекций предметов у рабочего места кассира (по группам PROBES)."""
    if scene is None or perc is None or not getattr(perc, "times", None):
        return []
    area = _work_area(scene)
    if area is None:
        return []
    times = np.asarray(perc.times, float)
    nfr = len(times)
    roles = scene.roles or {}
    cls2probe = {c: p.key for p in PROBES.values() for c in p.classes}
    raw: dict[str, list[tuple[float, tuple, float]]] = {k: [] for k in PROBES}
    for j, objs in enumerate(perc.objects):
        persons = [o for o in perc.persons[j] if o.inside] if j < len(perc.persons) else []
        cashier_boxes = [o.xyxy for o in persons if roles.get(o.track_id) == "cashier"]
        visitor_boxes = [o.xyxy for o in persons if roles.get(o.track_id) == "visitor"]
        for d in objs:
            key = cls2probe.get(d.cls)
            if key is None:
                continue
            c = _center(d.xyxy)
            with_cashier = any(_inside(c, cb, 0.1) for cb in cashier_boxes)
            at_desk = _inside(c, area)
            with_visitor = any(_inside(c, vb, 0.05) for vb in visitor_boxes)
            if not (with_cashier or (at_desk and not with_visitor)):
                continue
            raw[key].append((float(times[j]), tuple(float(v) for v in d.xyxy), float(d.score)))
    # статичный интерьер: та же точка видна в большей части ролика
    all_by_key: dict[str, list[tuple[float, tuple]]] = {k: [] for k in PROBES}
    for j, objs in enumerate(perc.objects):
        for d in objs:
            key = cls2probe.get(d.cls)
            if key is not None:
                all_by_key[key].append((float(times[j]), tuple(float(v) for v in d.xyxy)))
    out: list[Proposal] = []
    for key, dets in raw.items():
        dets = [x for x in dets if _persistence(x[1], all_by_key[key], nfr) < PROP_STATIC_FRAC]
        dets.sort(key=lambda x: x[0])
        cur: Optional[Proposal] = None
        for t, box, s in dets:
            if cur is not None and t - cur.times[-1] <= PROP_GAP_SEC:
                cur.times.append(t); cur.boxes.append(box); cur.scores.append(s)
                continue
            if cur is not None and cur.n >= PROP_MIN_DET and cur.t1 - cur.t0 >= PROP_MIN_SPAN_SEC:
                out.append(cur)
            cur = Proposal(probe=key, times=[t], boxes=[box], scores=[s])
        if cur is not None and cur.n >= PROP_MIN_DET and cur.t1 - cur.t0 >= PROP_MIN_SPAN_SEC:
            out.append(cur)
    # сильнее и раньше — первыми; бюджет — MAX_PROPOSALS
    out.sort(key=lambda p: (-p.n * max(p.scores), p.t0))
    return out[:MAX_PROPOSALS]


def _persistence(box, dets: list[tuple[float, tuple]], nfr: int) -> float:
    if nfr <= 0:
        return 0.0
    ts = {t for t, b in dets if _iou(box, b) >= PROP_STATIC_IOU}
    return len(ts) / float(nfr)


# ---------------------------------------------------------------------------
# проверка VLM
# ---------------------------------------------------------------------------

class AnswerCache:
    """Ответы VLM по ключу вопроса (только при FEATURES_CACHE): results/safety_net/<видео>.json."""

    def __init__(self, video_path: str) -> None:
        self.path: Optional[Path] = None
        self.data: dict[str, Optional[bool]] = {}
        self.dirty = False
        root = os.environ.get("FEATURES_CACHE")
        if not root:
            return
        stem = Path(video_path).stem.replace("X", "Х")
        self.path = Path(root) / "safety_net" / f"{stem}.json"
        try:
            if self.path.exists():
                d = json.loads(self.path.read_text(encoding="utf-8"))
                self.data = {k: v for k, v in (d.get("answers") or {}).items() if v in (True, False, None)}
        except Exception as exc:  # noqa: BLE001
            log.warning("safety_net: кеш %s не прочитан: %s", self.path, exc)

    def save(self, model: str = "") -> None:
        if self.path is None or not self.dirty:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({"model": model, "answers": self.data}, ensure_ascii=False, indent=0), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            log.warning("safety_net: кеш %s не записан: %s", self.path, exc)


class _Asker:
    """Обёртка: кеш → VLM, счётчик вызовов, бюджет времени."""

    def __init__(self, verifier: Optional[Verifier], cache: AnswerCache, budget_sec: float,
                 read_frame: Callable[[float], Optional[np.ndarray]]) -> None:
        self.verifier = verifier
        self.cache = cache
        self.deadline = time.monotonic() + budget_sec
        self.read_frame = read_frame
        self.calls = 0
        self.missing = 0
        self._frames: dict[float, Optional[np.ndarray]] = {}

    def out_of_time(self) -> bool:
        return time.monotonic() > self.deadline

    def ask(self, t: float, box, question: str) -> Optional[bool]:
        key = f"{t:.2f}|{','.join(v if isinstance(v, str) else str(int(round(v))) for v in box)}|{question}"
        if key in self.cache.data:
            return self.cache.data[key]
        if self.verifier is None or self.out_of_time():
            self.missing += 1
            return None
        if t not in self._frames:
            self._frames[t] = self.read_frame(t)
        frame = self._frames[t]
        if frame is None:
            return None
        ans = self.verifier.ask_yes_no(_crop(frame, box), question + _ANSWER)
        self.calls += 1
        self.cache.data[key] = ans
        self.cache.dirty = True
        return ans


def _pick_times(p: Proposal, duration: float) -> list[float]:
    """До VERIFY_MAX_FRAMES моментов эпизода, разнесённых ≥ VERIFY_MIN_GAP_SEC (сначала самые уверенные детекции)."""
    order = sorted(range(p.n), key=lambda k: -p.scores[k])
    chosen: list[float] = []
    for k in order:
        t = p.times[k]
        if all(abs(t - c) >= VERIFY_MIN_GAP_SEC for c in chosen):
            chosen.append(t)
        if len(chosen) >= VERIFY_MAX_FRAMES:
            break
    if len(chosen) < VERIFY_MIN_YES:   # короткий эпизод: предмет обычно ещё на месте через 1 с
        t = min(p.t1 + VERIFY_MIN_GAP_SEC, max(0.0, duration - 0.1))
        if all(abs(t - c) >= VERIFY_MIN_GAP_SEC for c in chosen):
            chosen.append(t)
    return sorted(chosen)


def _control_time(p: Proposal, perc, duration: float, roles: Optional[dict] = None) -> Optional[float]:
    """Контрольный момент: детектор не видел предмет этой группы в этой точке ±CONTROL_CLEAR_SEC, а кассир при
    этом в кадре и его рамка заходит во фрагмент (руки/стол на месте — «нет» на пустой участок ничего не доказывает).
    Самый далёкий от эпизода (та же запись, тот же ракурс). None — такого момента нет (тогда событие не добавляем)."""
    probe = PROBES[p.probe]
    box = p.box
    cb = crop_box(box, perc.info.width, perc.info.height)
    seen = [float(t) for t, objs in zip(perc.times, perc.objects)
            if any(d.cls in probe.classes and _iou(d.xyxy, box) >= 0.1 for d in objs)]
    seen += p.times
    cands = []
    for j, t in enumerate(perc.times):
        t = float(t)
        if not (0.5 <= t <= duration - 0.5) or any(abs(t - s) <= CONTROL_CLEAR_SEC for s in seen):
            continue
        if roles is not None:
            ps = perc.persons[j] if j < len(perc.persons) else []
            if not any(o.inside and roles.get(o.track_id) == "cashier" and _iou(o.xyxy, cb) > 0.0 for o in ps):
                continue
        cands.append(t)
    if not cands:
        return None
    return max(cands, key=lambda t: min(abs(t - p.t0), abs(t - p.t1)))


def verify_proposal(p: Proposal, asker: _Asker, perc, duration: float, roles: Optional[dict] = None) -> dict:
    """Проверка эпизода. Возвращает отчёт {"ok": bool, ...}; ok=True — событие можно добавлять."""
    probe = PROBES[p.probe]
    rep: dict[str, Any] = {"probe": p.probe, "t0": round(p.t0, 2), "t1": round(p.t1, 2), "n_det": p.n,
                           "max_score": round(max(p.scores), 2), "ok": False}
    times = _pick_times(p, duration)
    yes_times = []
    for t in times:
        k = int(np.argmin([abs(t - x) for x in p.times]))
        box = p.boxes[k] if abs(p.times[k] - t) < 0.4 else p.box
        if asker.ask(t, box, probe.question) is not True:
            continue
        if any(asker.ask(t, box, q) is True for q in probe.refute):
            rep.setdefault("refuted_at", []).append(round(t, 2))
            continue
        yes_times.append(t)
    rep["yes"] = [round(t, 2) for t in yes_times]
    rep["asked"] = [round(t, 2) for t in times]
    spaced = len(yes_times) >= VERIFY_MIN_YES and (max(yes_times) - min(yes_times)) >= VERIFY_MIN_GAP_SEC
    if not spaced:
        rep["reason"] = "vlm_not_confirmed"
        return rep
    tc = _control_time(p, perc, duration, roles)
    if tc is None:
        rep["reason"] = "no_control_frame"
        return rep
    ctrl = asker.ask(tc, p.box, probe.question)
    rep["control"] = {"t": round(tc, 2), "answer": ctrl}
    if ctrl is not False:
        rep["reason"] = "control_failed"
        return rep
    rep["ok"] = True
    return rep


def verify_on_desk(p: Proposal, asker: _Asker, scene, duration: float) -> dict:
    """«Куда делся внесённый предмет»: личную сумку внесли (детектор видел её у кассира), а потом она лежит на
    столешнице. Кадры столешницы после эпизода с шагом FOLLOW_STEP_SEC; нужно «да» на двух кадрах подряд, «нет» на
    опровержениях и «нет» по той же столешнице ДО появления предмета (контроль)."""
    probe = PROBES[p.probe]
    rep: dict[str, Any] = {"probe": p.probe + "_on_desk", "t0": round(p.t0, 2), "t1": round(p.t1, 2), "n_det": p.n, "ok": False}
    area = _desk_area_box(scene)
    if area is None or not probe.desk_question:
        rep["reason"] = "no desk"
        return rep
    ctrl_t = p.t0 - CONTROL_CLEAR_SEC
    if ctrl_t < 0.5:
        rep["reason"] = "no_control_frame"
        return rep
    times = [p.t1 + FOLLOW_START_SEC + k * FOLLOW_STEP_SEC for k in range(FOLLOW_MAX_FRAMES)]
    times = [t for t in times if t <= duration - 0.2]
    prev_yes: Optional[float] = None
    first: Optional[float] = None
    asked = []
    for t in times:
        if asker.out_of_time():
            break
        asked.append(round(t, 2))
        ok = asker.ask(t, area, probe.desk_question) is True and not any(asker.ask(t, area, q) is True for q in probe.refute)
        if ok and prev_yes is not None:
            first = prev_yes
            break
        prev_yes = t if ok else None
    rep["asked"] = asked
    if first is None:
        rep["reason"] = "vlm_not_confirmed"
        return rep
    rep["yes"] = [round(first, 2), round(first + FOLLOW_STEP_SEC, 2)]
    ctrl = asker.ask(ctrl_t, area, probe.desk_question)
    rep["control"] = {"t": round(ctrl_t, 2), "answer": ctrl}
    if ctrl is not False:
        rep["reason"] = "control_failed"
        return rep
    rep["ok"] = True
    rep["t_on_desk"] = round(first, 2)
    rep["area"] = [round(v, 1) for v in area[:4]]
    return rep


def _desk_event(p: Proposal, rep: dict) -> Event:
    probe = PROBES[p.probe]
    t0 = float(rep["t_on_desk"])
    desc = (
        f"В {p.t0:.1f}–{p.t1:.1f} с кассир вносит в кассу сумку (детектор фиксирует её у кассира на {p.n} кадрах), "
        f"с {t0:.1f} с на рабочем столе лежит личная сумка: локальная модель подтверждает это на двух кадрах подряд "
        f"и не видит сумки на столе до её появления ({rep['control']['t']:.1f} с). Может указывать на внос личных "
        f"вещей в кассовый узел (п. 12 Правил № 231: хранение личных вещей работников — вне кассового узла). "
        f"Принадлежность сумки по видео не устанавливается. Событие найдено страховочным слоем и требует проверки человеком."
    )
    box = tuple(int(v) for v in rep["area"])
    ev_times = [t0, t0 + FOLLOW_STEP_SEC, float(p.t0)]
    return Event(
        type_code=probe.type_code, timecode_sec=round(t0, 2), confidence=0.6, description=desc,
        evidence_times=ev_times, evidence_captions=[probe.caption_ru, probe.caption_ru, "кассир вносит сумку"],
        extras={"source": SOURCE, "probe": p.probe + "_on_desk", "evidence_boxes": [box, box, tuple(int(v) for v in p.box)], "verify": rep},
    )


def _proposal_event(p: Proposal, rep: dict) -> Event:
    probe = PROBES[p.probe]
    t0 = float(p.t0)
    yes = rep.get("yes") or [t0]
    desc = (
        f"С {t0:.1f} с у рабочего места кассира видна {probe.what_ru}: детектор фиксирует предмет на "
        f"{p.n} кадрах ({p.t0:.1f}–{p.t1:.1f} с), локальная модель подтверждает это по увеличенному фрагменту "
        f"на {len(yes)} кадрах и не видит предмета в этой точке в контрольный момент {rep['control']['t']:.1f} с. "
        f"Может указывать на нахождение личных вещей на рабочем месте в кассовом узле (п. 12 Правил № 231: хранение "
        f"личных вещей работников — вне кассового узла). {probe.hedge_ru + ' ' if probe.hedge_ru else ''}"
        f"Событие найдено страховочным слоем и требует проверки человеком."
    )
    box = tuple(int(v) for v in p.box)
    ev_times = [t0] + [t for t in yes if abs(t - t0) >= 0.5][:2]
    return Event(
        type_code=probe.type_code, timecode_sec=round(t0, 2), end_sec=round(float(p.t1), 2), confidence=0.6,
        description=desc, evidence_times=ev_times,
        evidence_captions=[probe.caption_ru] * len(ev_times),
        extras={"source": SOURCE, "probe": p.probe, "evidence_boxes": [box] * len(ev_times), "verify": rep},
    )


# ---------------------------------------------------------------------------
# точка входа
# ---------------------------------------------------------------------------

def _vlm_verifier(report: NetReport) -> Optional[Verifier]:
    """VLM, если включена, загружена и достаточно быстрая на этой машине; иначе None (с причиной в отчёте)."""
    try:
        from .vlm import get_describer

        d = get_describer()
        if not d.available():
            report.vlm = "unavailable"
            report.reason = d.unavailable_reason()
            return None
        if not d.load():
            report.vlm = "unavailable"
            report.reason = d.unavailable_reason() or "load failed"
            return None
        if not d.fast_enough():
            report.vlm = "slow"
            report.reason = f"latency {d.latency_sec}s > limit"
            return None
        report.vlm = "ready"
        return d
    except Exception as exc:  # noqa: BLE001
        report.vlm = "unavailable"
        report.reason = f"{type(exc).__name__}: {exc}"
        return None


def run_safety_net(ctx: VideoContext, events: list[Event], perc=None) -> tuple[list[Event], NetReport]:
    """Добавочные события страховочной сетки. Никогда не бросает исключений и не меняет `events`."""
    report = NetReport()
    t_start = time.monotonic()
    mode = _env("SAFETY_NET", "auto")
    if mode in {"0", "off", "false", "no"}:
        report.reason = "disabled"
        return [], report
    report.enabled = True
    added: list[Event] = []
    scene = getattr(ctx, "scene", None)
    try:
        if _env("SAFETY_NET_STRUCTURAL", "1") not in {"0", "off", "false", "no"}:
            added.extend(structural_events(scene, events))
    except Exception as exc:  # noqa: BLE001
        log.exception("safety_net: структурный слой упал: %s", exc)

    try:
        has_rule_violation = bool(events) or bool(added)
        if has_rule_violation and _env("SAFETY_NET_ALWAYS", "0") not in {"1", "true", "yes", "on"}:
            report.reason = report.reason or "rules already found a violation"
            return added, _finish(report, t_start)
        if scene is None or perc is None:
            report.reason = "no perception"
            return added, _finish(report, t_start)
        proposals = item_proposals(scene, perc)
        report.proposals = len(proposals)
        if not proposals:
            report.reason = report.reason or "no proposals"
            return added, _finish(report, t_start)
        cache = AnswerCache(ctx.path)
        verifier = _vlm_verifier(report)
        if verifier is None:
            if mode in {"1", "on", "true", "yes"} and cache.data:
                report.vlm = "cache"
            else:
                return added, _finish(report, t_start)
        from .video import read_frame_at

        budget = _env_float("SAFETY_NET_BUDGET_SEC", DEFAULT_BUDGET_SEC)
        from . import budget as _budget

        bud = _budget.current()
        if bud is not None:   # общий бюджет 5 минут: сетка не съедает запас, нужный на кадры-доказательства
            budget = min(budget, max(0.0, bud.left_hard() - _budget.MIN_VLM_LEFT_SEC))
        asker = _Asker(verifier, cache, budget, lambda t: read_frame_at(ctx.path, t, ctx.info))
        covered = {e.type_code for e in events + added}
        for p in proposals:
            if asker.out_of_time():
                report.details.append({"probe": p.probe, "t0": round(p.t0, 2), "reason": "budget"})
                break
            probe = PROBES[p.probe]
            if probe.type_code in covered:
                continue
            rep = verify_proposal(p, asker, perc, ctx.info.duration_sec, scene.roles)
            report.details.append(rep)
            if rep["ok"]:
                added.append(_proposal_event(p, rep))
                covered.add(probe.type_code)
                report.verified += 1
                continue
            if probe.follow_desk and rep.get("reason") == "vlm_not_confirmed" and not rep.get("refuted_at"):
                rep2 = verify_on_desk(p, asker, scene, ctx.info.duration_sec)
                report.details.append(rep2)
                if rep2["ok"]:
                    added.append(_desk_event(p, rep2))
                    covered.add(probe.type_code)
                    report.verified += 1
        report.vlm_calls = asker.calls
        model = getattr(verifier, "model_name", "") if verifier is not None else ""
        cache.save(model)
    except Exception as exc:  # noqa: BLE001 — сетка необязательна, анализ не должен падать
        log.exception("safety_net: предметный слой упал: %s", exc)
        report.reason = f"error: {type(exc).__name__}: {exc}"
    return added, _finish(report, t_start)


def _finish(report: NetReport, t_start: float) -> NetReport:
    report.elapsed_sec = time.monotonic() - t_start
    return report


__all__ = ["run_safety_net", "structural_events", "item_proposals", "verify_proposal", "PROBES", "Proposal", "NetReport"]
