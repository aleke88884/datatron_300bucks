"""Справочник правил и реестр детекторов: нет выдуманных пунктов, все T1–T13 на месте."""
from __future__ import annotations

import re
import sys

import pytest

from .conftest import SERVICE_DIR

if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

from pipeline.detectors import DETECTORS  # noqa: E402
from pipeline.rules import CLAUSES, DOCUMENTS, RULES, api_rule  # noqa: E402

ALL_CODES = [f"T{i}" for i in range(1, 14)]


def test_all_types_present():
    assert list(RULES) == ALL_CODES


@pytest.mark.parametrize("code", ALL_CODES)
def test_rule_entry_is_complete(code):
    r = RULES[code]
    assert r["code"] == code
    assert r["name"].strip() and r["sign"].strip()
    assert isinstance(r["examples"], list) and r["examples"]
    for ref in [r["rule"], *r["secondary"]]:
        assert ref["document"] and ref["clause"] and ref["url"]
        assert ref["url"] in {d["url"] for d in DOCUMENTS.values()}, "только adilet-ссылки на два постановления НБРК"


def _doc_key_by_url(url: str) -> str:
    return next(k for k, d in DOCUMENTS.items() if d["url"] == url)


@pytest.mark.parametrize("code", ALL_CODES)
def test_clauses_exist_in_source_tables(code):
    """Каждый пункт из `clause` должен быть в таблице раздела 4 плана — защита от выдуманных ссылок (критерий 4 ТЗ)."""
    r = RULES[code]
    for ref in [r["rule"], *r["secondary"]]:
        doc = _doc_key_by_url(ref["url"])
        known = CLAUSES[doc]
        parts = [p.strip() for p in ref["clause"].split(",")]
        # «п. 16 пп. 3), пп. 4)» → второй элемент наследует номер пункта
        expanded = []
        base = None
        for p in parts:
            m = re.match(r"^(п\. [\d–-]+)", p)
            if m:
                base = m.group(1)
                expanded.append(p)
            else:
                assert base and p.startswith("пп."), p
                expanded.append(f"{base} {p}")
        for p in expanded:
            assert p in known, f"{code}: пункт «{p}» отсутствует в таблице {doc}"


def test_documents_have_registration_and_urls():
    for key, d in DOCUMENTS.items():
        assert d["url"].startswith("https://adilet.zan.kz/rus/docs/V")
        assert "Национального Банка" in d["title"] and "№" in d["title"]
    assert "приостановлено" in DOCUMENTS["R14"]["note"]


@pytest.mark.parametrize("code", ALL_CODES)
def test_api_rule_shape(code):
    r = api_rule(code)
    assert set(r) >= {"document", "clause", "url"}
    assert set(r) <= {"document", "clause", "url", "note"}


def test_detectors_cover_every_type_once():
    codes = []
    for fn in DETECTORS:
        m = re.search(r"_t(\d{1,2})_", fn.__name__)
        assert m, fn.__name__
        codes.append(f"T{m.group(1)}")
        assert fn.__doc__ and fn.__doc__.strip().startswith(f"{codes[-1]}."), f"{fn.__name__}: докстринг должен начинаться с кода"
    assert sorted(codes, key=lambda c: int(c[1:])) == ALL_CODES
    assert len(set(codes)) == len(codes)


def test_stub_detectors_return_empty_lists():
    """Заглушки не должны порождать событий (иначе — ложные срабатывания на чистых видео)."""
    import numpy as np

    from pipeline.context import VideoContext
    from pipeline.video import VideoInfo

    info = VideoInfo(path="x", width=464, height=848, fps=30.0, frame_count=300, duration_sec=10.0)
    ctx = VideoContext(path="x", info=info, bright_t=np.linspace(0, 10, 100), bright_v=np.full(100, 130.0))
    for fn in DETECTORS:
        assert fn(ctx) == [], fn.__name__


def test_context_only_types_are_consistent():
    """Контекстные типы (не дают вердикт) помечены и в реестре правил, и в пайплайне одинаково."""
    from pipeline import CONTEXT_ONLY_TYPES

    flagged = {c for c, r in RULES.items() if r.get("context_only")}
    assert flagged == set(CONTEXT_ONLY_TYPES) == {"T9"}
    assert RULES["T9"]["caveat"]
