"""Фронтенд: страница отдаётся с корня, на русском, содержит все элементы сценария ТЗ и ходит в нужные эндпоинты."""
from __future__ import annotations

import re

import pytest

REQUIRED_TEXT = [
    "Контроль кассовой дисциплины",
    "X-API-Key",
    "Перетащите видео",
    "Отправить на анализ",
    "Скачать JSON",
    "Нарушение выявлено",
    "Нарушений не выявлено",
    "Нормативное основание",
    "adilet",
    "без распознавания лиц",
    "Контекст сцены",
]


@pytest.fixture(scope="module")
def page(server) -> str:
    r = server.get("/", key=None)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    return r.text


def test_page_is_russian_and_has_all_elements(page):
    assert 'lang="ru"' in page
    for text in REQUIRED_TEXT:
        assert text in page, text


def test_page_has_upload_controls(page):
    assert re.search(r'<input[^>]+type="file"[^>]+accept="[^"]*\.mov', page)
    assert 'id="drop"' in page and "dragover" in page and "'drop'" in page
    assert re.search(r'<input[^>]+id="apikey"', page)


def test_page_calls_tz_endpoints_with_key_header(page):
    assert "/api/v1/analyze" in page
    assert "/api/v1/jobs/" in page
    assert "'X-API-Key'" in page
    assert "image/jpeg;base64" in page, "кадры показываются из image_base64"
    assert "context_events" in page, "контекстные события из meta показываются отдельно"


def test_page_handles_all_statuses(page):
    for s in ("'processing'", "'done'", "401", "413", "415"):
        assert s in page, s


def test_static_mount(server):
    r = server.get("/static/index.html", key=None)
    assert r.status_code == 200 and "Контроль кассовой дисциплины" in r.text
    assert server.get("/static/nope.html", key=None).status_code == 404
