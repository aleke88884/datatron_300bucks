"""VLM-слой без модели: модуль импортируется без torch/transformers, при выключенном слое ничего не меняет и не падает."""
from __future__ import annotations

import importlib
import sys

import numpy as np
import pytest

from .conftest import SERVICE_DIR

if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

import pipeline.vlm as vlm  # noqa: E402
from pipeline.rules import RULES  # noqa: E402

HEAVY_MODULES = ("torch", "transformers", "accelerate", "bitsandbytes", "qwen_vl_utils")


def _frame(h: int = 72, w: int = 128) -> np.ndarray:
    return np.full((h, w, 3), 127, dtype=np.uint8)


def _violations() -> list[dict]:
    return [
        {
            "type": RULES["T1"]["name"],
            "type_code": "T1",
            "timecode_sec": 12.5,
            "description": "Кассир вне зоны стола дольше 10 с, на столе остаются наличные без присмотра.",
            "evidence_frames": [{"timestamp_sec": 12.5, "image_base64": ""}, {"timestamp_sec": 13.5, "image_base64": ""}],
            "confidence": 0.8,
            "rule": {"document": "d", "clause": "п. 20", "url": "https://adilet.zan.kz/rus/docs/V1900019680"},
        },
        {
            "type": RULES["T2"]["name"],
            "type_code": "T2",
            "timecode_sec": 30.0,
            "description": "Дверца сейфа открыта, кассир вышел из помещения.",
            "evidence_frames": [{"timestamp_sec": 30.0, "image_base64": ""}],
            "confidence": 0.7,
            "rule": {"document": "d", "clause": "п. 74", "url": "https://adilet.zan.kz/rus/docs/V1900019680"},
        },
    ]


@pytest.fixture
def no_vlm_env(monkeypatch):
    """Слой выключен, никаких весов, свежий синглтон."""
    for k in ("VLM_ENABLED", "VLM_MODEL_PATH", "VLM_4BIT"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(vlm, "_describer", None)
    yield
    monkeypatch.setattr(vlm, "_describer", None)


@pytest.fixture
def fake_weights(tmp_path, monkeypatch):
    """Папка «с весами» (config.json + пустой safetensors), чтобы available() стал True при VLM_ENABLED=1."""
    d = tmp_path / "vlm"
    d.mkdir()
    (d / "config.json").write_text("{}", encoding="utf-8")
    (d / "model-00001-of-00001.safetensors").write_bytes(b"")
    monkeypatch.setenv("VLM_MODEL_PATH", str(d))
    monkeypatch.setattr(vlm, "_describer", None)
    yield d
    monkeypatch.setattr(vlm, "_describer", None)


# ----------------------------------------------------------------------------- импорт и доступность


def test_module_imports_without_torch_and_transformers(monkeypatch):
    """pipeline.vlm должен импортироваться там, где torch/transformers не установлены (sys.modules[...] = None →
    ImportError на любой import). Тяжёлые пакеты импортируются только внутри Describer.load()."""
    for name in HEAVY_MODULES:
        monkeypatch.setitem(sys.modules, name, None)
    mod = importlib.reload(vlm)
    assert hasattr(mod, "Describer") and hasattr(mod, "describe_violations")
    for name in HEAVY_MODULES:
        monkeypatch.delitem(sys.modules, name, raising=False)
    importlib.reload(vlm)


def test_load_without_torch_returns_false_not_raises(monkeypatch, fake_weights):
    monkeypatch.setenv("VLM_ENABLED", "1")
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "transformers", None)
    d = vlm.Describer()
    assert d.available() is True
    assert d.load() is False
    assert d._load_error and "torch" in d._load_error  # ModuleNotFoundError (подкласс ImportError)
    assert d.available() is False, "после неудачной загрузки слой считается недоступным (без повторных попыток)"


def test_not_available_without_env(no_vlm_env, fake_weights):
    assert vlm.Describer().available() is False


@pytest.mark.parametrize("value", ["0", "false", "no", ""])
def test_not_available_with_falsy_env(monkeypatch, fake_weights, value):
    monkeypatch.setenv("VLM_ENABLED", value)
    assert vlm.Describer().available() is False


def test_not_available_without_weights(monkeypatch, tmp_path):
    monkeypatch.setenv("VLM_ENABLED", "1")
    monkeypatch.setenv("VLM_MODEL_PATH", str(tmp_path / "missing"))
    assert vlm.Describer().available() is False
    (tmp_path / "missing").mkdir()
    (tmp_path / "missing" / "config.json").write_text("{}", encoding="utf-8")
    assert vlm.Describer().available() is False, "config.json без *.safetensors — весов нет"


def test_available_with_env_and_weights(monkeypatch, fake_weights):
    monkeypatch.setenv("VLM_ENABLED", "1")
    assert vlm.Describer().available() is True
    assert vlm.Describer().model_path == fake_weights


# ----------------------------------------------------------------------------- describe_violations без модели


def test_describe_violations_disabled_keeps_descriptions(no_vlm_env):
    vs = _violations()
    before = [dict(v) for v in vs]
    vlm.describe_violations("/nonexistent/video.mov", vs, None)
    assert vs == before
    assert all("description_template" not in v for v in vs)


def test_describe_violations_enabled_but_load_fails_keeps_descriptions(monkeypatch, fake_weights):
    monkeypatch.setenv("VLM_ENABLED", "1")
    monkeypatch.setattr(vlm.Describer, "_load_unlocked", lambda self: (_ for _ in ()).throw(RuntimeError("no GPU")))
    vs = _violations()
    before = [dict(v) for v in vs]
    vlm.describe_violations("/nonexistent/video.mov", vs, None)  # видео тоже нет — и это не должно падать
    assert vs == before


def test_describe_violations_empty_list_and_bad_input(no_vlm_env):
    vlm.describe_violations("/nonexistent/video.mov", [], None)
    vlm.describe_violations("/nonexistent/video.mov", [{"type_code": "T1"}], None)  # без description/evidence


def test_describe_event_unavailable_returns_none(no_vlm_env):
    assert vlm.Describer().describe_event([_frame()], _violations()[0], RULES["T1"]) is None


# ----------------------------------------------------------------------------- подмена описаний с «моделью-заглушкой»


def _fake_loaded(monkeypatch, answer):
    """Описатель, у которого «модель загружена», а _generate возвращает answer (или вызывает его)."""
    def _load(self):
        self.model = self.processor = object()
        self.device = "cpu"

    monkeypatch.setattr(vlm.Describer, "_load_unlocked", _load)

    calls: list[str] = []

    def _gen(self, images, user_text, max_new_tokens, timeout_sec):
        calls.append(user_text)
        return answer(user_text) if callable(answer) else answer

    monkeypatch.setattr(vlm.Describer, "_generate", _gen)
    return calls


def test_describe_violations_replaces_description_with_stub_model(monkeypatch, fake_weights):
    monkeypatch.setenv("VLM_ENABLED", "1")
    text = "Кассир стоит у двери спиной к столу. На столе лежит пачка купюр, рядом никого нет. Дверца сейфа закрыта."
    calls = _fake_loaded(monkeypatch, text)
    monkeypatch.setattr("pipeline.video.read_frame_at", lambda path, t, info=None: _frame())
    vs = _violations()
    templates = [v["description"] for v in vs]
    vlm.describe_violations("/fake.mov", vs, None)
    assert [v["description"] for v in vs] == [text, text]
    assert [v["description_template"] for v in vs] == templates
    assert len(calls) == 2
    assert RULES["T1"]["name"] in calls[0] and templates[0] in calls[0], "в запросе есть тип события и шаблон"
    assert "похитил" in calls[0] and "внешность" in calls[0], "запреты на юридические выводы и внешность — в запросе"


def test_describe_violations_respects_max_events(monkeypatch, fake_weights):
    monkeypatch.setenv("VLM_ENABLED", "1")
    monkeypatch.setenv("VLM_MAX_EVENTS", "1")
    text = "Кассир отошёл от стола к двери. На столе остаются наличные, других людей в помещении нет."
    calls = _fake_loaded(monkeypatch, text)
    monkeypatch.setattr("pipeline.video.read_frame_at", lambda path, t, info=None: _frame())
    vs = _violations()
    vlm.describe_violations("/fake.mov", vs, None)
    assert len(calls) == 1
    assert vs[0]["description"] == text
    assert "description_template" not in vs[1]


def test_describe_violations_skips_when_frames_unreadable(monkeypatch, fake_weights):
    monkeypatch.setenv("VLM_ENABLED", "1")
    calls = _fake_loaded(monkeypatch, "Кассир отошёл от стола, на столе остаются наличные, сейф закрыт.")
    monkeypatch.setattr("pipeline.video.read_frame_at", lambda path, t, info=None: None)
    vs = _violations()
    before = [dict(v) for v in vs]
    vlm.describe_violations("/fake.mov", vs, None)
    assert vs == before and calls == []


def test_rejected_answer_keeps_template(monkeypatch, fake_weights):
    monkeypatch.setenv("VLM_ENABLED", "1")
    _fake_loaded(monkeypatch, "Кассир похитил деньги со стола и спрятал их в карман, это мошенничество.")
    monkeypatch.setattr("pipeline.video.read_frame_at", lambda path, t, info=None: _frame())
    vs = _violations()
    before = [dict(v) for v in vs]
    vlm.describe_violations("/fake.mov", vs, None)
    assert vs == before


def test_generate_exception_is_swallowed(monkeypatch, fake_weights):
    monkeypatch.setenv("VLM_ENABLED", "1")

    def _boom(user_text):
        raise RuntimeError("CUDA out of memory")

    _fake_loaded(monkeypatch, _boom)
    monkeypatch.setattr("pipeline.video.read_frame_at", lambda path, t, info=None: _frame())
    vs = _violations()
    before = [dict(v) for v in vs]
    vlm.describe_violations("/fake.mov", vs, None)
    assert vs == before


# ----------------------------------------------------------------------------- checklist


def test_checklist_unavailable_returns_none_for_each_question(no_vlm_env):
    qs = ["На столе кассира есть стакан или еда?", "В помещении больше двух человек?"]
    assert vlm.Describer().checklist([_frame()], qs) == {q: None for q in qs}
    assert vlm.Describer().checklist([_frame()], []) == {}


def test_checklist_parses_yes_no_with_stub_model(monkeypatch, fake_weights):
    monkeypatch.setenv("VLM_ENABLED", "1")
    answers = {"стакан": "Да.", "человек": "нет", "сейф": "Не могу сказать"}
    _fake_loaded(monkeypatch, lambda user_text: next(a for k, a in answers.items() if k in user_text))
    qs = ["На столе кассира есть стакан или еда?", "В помещении больше двух человек?", "Дверца сейфа открыта?"]
    assert vlm.Describer().checklist([_frame()], qs) == {qs[0]: True, qs[1]: False, qs[2]: None}


@pytest.mark.parametrize(
    "raw,expected",
    [("Да", True), ("да, видно стакан", True), ("Yes", True), ("Нет.", False), ("no", False),
     ("Неизвестно", None), ("", None), (None, None), ("Возможно, да", None)],
)
def test_parse_yes_no(raw, expected):
    assert vlm.parse_yes_no(raw) is expected


# ----------------------------------------------------------------------------- постобработка описания


def test_sanitize_accepts_neutral_russian_text():
    t = "Кассир стоит у двери, спиной к столу. На столе лежит пачка купюр. Дверца сейфа закрыта."
    assert vlm.sanitize_description(t) == t


def test_sanitize_strips_wrappers_and_limits_sentences():
    raw = "Описание: «Кассир у двери. На столе купюры. Сейф закрыт. Экран ноутбука активен. Клиента нет.»"
    out = vlm.sanitize_description(raw)
    assert out == "Кассир у двери. На столе купюры. Сейф закрыт."


@pytest.mark.parametrize(
    "raw",
    [
        None, "", "   ", "Коротко.",
        "Кассир украл деньги со стола и ушёл из помещения.",
        "На кадрах видно, что сотрудник совершил хищение наличных.",
        "Это явное мошенничество со стороны кассира в кассовом помещении.",
        "Мужчина в очках стоит у стола, на столе лежат деньги.",
        "The cashier left the desk and the cash remains on the table unattended.",
        "出纳员离开了桌子，现金留在桌子上。",
    ],
)
def test_sanitize_rejects_bad_answers(raw):
    assert vlm.sanitize_description(raw) is None


def test_sanitize_rejects_verbatim_template():
    t = "Кассир вне зоны стола дольше 10 с, на столе остаются наличные без присмотра."
    assert vlm.sanitize_description(t, template=t) is None
    assert vlm.sanitize_description(t, template="другой шаблон") == t
