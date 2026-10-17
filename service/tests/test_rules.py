"""Справочник правил и реестр детекторов: нет выдуманных пунктов, все T1–T13 на месте."""
from __future__ import annotations

import re
import sys

import pytest

from .conftest import SERVICE_DIR

if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

from pipeline.detectors import DETECTORS  # noqa: E402
from pipeline.rules import CLAUSES, DOCUMENTS, QUOTE_SEP, QUOTES, RULES, api_rule, expand_clause  # noqa: E402

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
        assert ref["url"] in {d["url"] for d in DOCUMENTS.values()}, "только adilet-ссылки на акты из DOCUMENTS"
        assert ref["quote"].strip(), "у каждого пункта есть дословная цитата"


def _doc_key_by_url(url: str) -> str:
    return next(k for k, d in DOCUMENTS.items() if d["url"] == url)


@pytest.mark.parametrize("code", ALL_CODES)
def test_clauses_exist_in_source_tables(code):
    """Каждый пункт из `clause` должен быть в таблице раздела 4 плана — защита от выдуманных ссылок (критерий 4 ТЗ)."""
    r = RULES[code]
    for ref in [r["rule"], *r["secondary"]]:
        doc = _doc_key_by_url(ref["url"])
        known = CLAUSES[doc]
        # «п. 16 пп. 3), пп. 4)» → второй элемент наследует номер пункта; «п. 20 абз. 2» — один ключ
        for p in expand_clause(ref["clause"]):
            assert p in known, f"{code}: пункт «{p}» отсутствует в таблице {doc}"


def test_documents_have_registration_and_urls():
    for key, d in DOCUMENTS.items():
        assert d["url"].startswith("https://adilet.zan.kz/rus/docs/V")
        assert ("Национального Банка" in d["title"] or "финансового рынка" in d["title"]) and "№" in d["title"]
        assert d["mirror"].endswith(d["url"].rsplit("/", 1)[1])
    assert "приостановлено" in DOCUMENTS["R14"]["note"]


@pytest.mark.parametrize("code", ALL_CODES)
def test_api_rule_shape(code):
    r = api_rule(code)
    assert set(r) >= {"document", "clause", "url"}
    assert set(r) <= {"document", "clause", "url", "quote", "note", "document_short", "clause_summaries"}
    assert r["quote"]


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", t)


def test_quotes_cover_clauses():
    """Каждому пункту из CLAUSES соответствует цитата, и наоборот; в цитате нет многоточий."""
    for doc, clauses in CLAUSES.items():
        assert set(clauses) == set(QUOTES[doc]), doc
        for k, q in QUOTES[doc].items():
            assert "…" not in q and "..." not in q, (doc, k)


@pytest.mark.parametrize("code", ALL_CODES)
def test_api_quote_is_joined_verbatim_quotes(code):
    r = RULES[code]["rule"]
    doc = _doc_key_by_url(r["url"])
    assert api_rule(code)["quote"] == QUOTE_SEP.join(QUOTES[doc][k] for k in expand_clause(r["clause"]))


def test_quotes_are_in_docs():
    """Каждая цитата дословно присутствует в docs/RULES_SOURCES.md (таблица «пункт → цитата»)."""
    docs = _norm((SERVICE_DIR.parent / "docs" / "RULES_SOURCES.md").read_text(encoding="utf-8"))
    for doc, qs in QUOTES.items():
        for k, q in qs.items():
            assert _norm(q) in docs, f"{doc} {k}: цитата отсутствует в docs/RULES_SOURCES.md"


def test_t3_cites_exact_it_security_clause():
    """FAQ 32: T3 привязан к точному пункту (п. 71 пп. 9) Требований ИБ), а не к «внутренним требованиям банка»."""
    r = RULES["T3"]["rule"]
    assert r["clause"] == "п. 71 пп. 9)" and r["url"].endswith("V2600038336")
    assert "30" in r["quote"] and "уточняется" not in r["note"]
    assert "не нарушает" in r["note"], "честно: автоблокировка по таймауту, а не в момент ухода"
    assert any(x["url"].endswith("V1800016772") for x in RULES["T3"]["secondary"]), "историческая редакция (НБРК № 48)"


def test_descriptions_are_observable_only():
    """ТЗ п. 7: в описаниях нет выводов о нарушении/блокировке — только наблюдаемое и пометка для проверки человеком."""
    import inspect

    from pipeline import detectors, rules_scene

    src = inspect.getsource(rules_scene) + inspect.getsource(detectors)
    for bad in ("не заблокировано", "должен быть исключён", "должен оставаться", "не допускаются", "должна оставаться",
                "должны храниться", "должны выполняться", "должны находиться", "не должна оставаться",
                "без предъявления документов"):
        assert bad not in src, f"вывод в описании: «{bad}»"


@pytest.mark.parametrize("code", ALL_CODES)
def test_api_rule_clause_summaries_come_from_clauses_table(code):
    """Краткое содержание пункта для UI берётся только из CLAUSES (без выдуманных пунктов) и покрывает весь clause."""
    r = api_rule(code)
    doc = next(k for k, d in DOCUMENTS.items() if d["short"] == r["document_short"])
    summ = r.get("clause_summaries") or []
    assert summ, f"{code}: нет краткого содержания для {r['clause']}"
    for item in summ:
        assert CLAUSES[doc][item["clause"]] == item["summary"]
    # каждый номер пункта из clause представлен
    for num in re.findall(r"(?<!п)п\. (\d+)", r["clause"]):
        assert any(item["clause"].startswith(f"п. {num}") for item in summ), (code, num)


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
