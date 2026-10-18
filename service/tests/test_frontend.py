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
    # формулировки вердикта — дословно по ТЗ
    "Нарушение не обнаружено",
    "Обнаружены признаки возможных нарушений",
    # дисклеймер: результат — подсказка для человека, не юридическое заключение
    "подсказка для проверки человеком",
    "не юридическое заключение",
    "Пункт для сверки",   # формулировка по ревью права (ПЛАН §14 п. 5): ссылка — для сверки, не вывод
    "Выдержка из текста пункта (дословно)",
    "признак нарушения",
    "adilet",
    "без распознавания лиц",
    "Контекст сцены",
    "JSON без кадров",
    "перейти к моменту",
    "Содержание пункта",
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
    for s in ("'processing'", "'done'", "401", "413", "415", "429", "status>=500", "timeout"):
        assert s in page, s


def test_page_has_player_and_timeline(page):
    """Плеер локального файла (без повторной загрузки на сервер), шкала времени с маркерами событий и дорожками."""
    assert re.search(r'<video[^>]+id="video"', page)
    assert "URL.createObjectURL(f)" in page and "URL.revokeObjectURL(videoUrl)" in page
    assert 'id="timeline"' in page and 'id="tlTracks"' in page and 'id="playhead"' in page
    assert "timeupdate" in page and "currentTime" in page
    assert "meta.timeline" in page or "m.timeline" in page
    for lane in ("cashier", "at_desk", "visitors", "safe_open", "safe_unknown", "desk_items", "screen_active",
                 "screen_unknown", "lights_off", "unreliable"):
        assert lane in page, lane


def test_page_shows_all_pipeline_stages(page):
    for stage in ("queued", "decoding", "detection", "states", "rules", "evidence", "describing"):
        assert re.search(rf"\b{stage}:'", page), stage


def test_page_is_offline_and_escapes_api_text(page):
    """Без внешних скриптов/стилей/шрифтов (работает без интернета); текст из API выводится через esc()."""
    assert not re.search(r'<(script|link)[^>]+(src|href)="https?://', page)
    assert "@import" not in page and "fonts.googleapis" not in page
    for field in ("esc(v.description)", "esc(r.note)", "esc(r.document)", "esc(s.summary)", "esc(e.description)"):
        assert field in page, field


def test_page_has_theme_and_mobile_support(page):
    assert 'name="viewport"' in page
    assert "prefers-color-scheme: dark" in page and 'data-theme="dark"' in page
    assert "@media (max-width: 640px)" in page
    assert "<dialog" in page and "ArrowRight" in page


def test_static_mount(server):
    r = server.get("/static/index.html", key=None)
    assert r.status_code == 200 and "Контроль кассовой дисциплины" in r.text
    assert server.get("/static/nope.html", key=None).status_code == 404
