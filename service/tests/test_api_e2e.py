"""E2E: настоящий сервер + настоящий пайплайн, запросы по HTTP как их будет слать жюри."""
from __future__ import annotations

import concurrent.futures as cf
import json
import os
import shutil
import subprocess
import time

import pytest

from .conftest import LONGEST_VIDEO, NEGATIVE_VIDEO, POSITIVE_VIDEO, TEST_API_KEY, video_path
from .schema import JOB_ID_RE, assert_done_result, assert_jpeg_base64

pytestmark = pytest.mark.dataset


# ---------------------------------------------------------------- здоровье и статика

def test_health_without_key(server):
    r = server.get("/health", key=None)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert set(["worker_alive", "queue", "jobs"]).issubset(body)


def test_openapi_lists_tz_endpoints(server):
    spec = server.get("/openapi.json", key=None).json()
    assert "/api/v1/analyze" in spec["paths"] and "post" in spec["paths"]["/api/v1/analyze"]
    assert "/api/v1/jobs/{job_id}" in spec["paths"] and "get" in spec["paths"]["/api/v1/jobs/{job_id}"]
    assert "/health" in spec["paths"]
    assert server.get("/docs", key=None).status_code == 200


# ---------------------------------------------------------------- авторизация

def test_web_ui_works_without_key_but_api_does_not(server):
    """Веб-интерфейс без авторизации (ПЛАН/FAQ): страница выдаёт HttpOnly-cookie; голый вызов API без ключа — 401."""
    import requests

    sess = requests.Session()
    page = sess.get(server.base_url + "/", timeout=30)
    assert page.status_code == 200
    assert "httponly" in page.headers.get("set-cookie", "").lower()
    assert sess.get(server.base_url + "/api/v1/jobs/deadbeef", timeout=30).status_code == 404   # cookie принята
    assert requests.get(server.base_url + "/api/v1/jobs/deadbeef", timeout=30).status_code == 401
    # неверный ключ не «спасается» cookie
    assert sess.get(server.base_url + "/api/v1/jobs/deadbeef", headers={"X-API-Key": "wrong"}, timeout=30).status_code == 401
    assert server.get("/health", key=None).json()["ui_open"] is True


@pytest.mark.parametrize("key", [None, "", "wrong-key", TEST_API_KEY + "x", TEST_API_KEY[:-1]])
def test_analyze_rejects_bad_key(server, negative_video, key):
    r = server.post_video(negative_video, key=key)
    assert r.status_code == 401, r.text
    assert "X-API-Key" in r.json()["detail"]


@pytest.mark.parametrize("path", ["/api/v1/jobs/deadbeef", "/api/v1/rules"])
@pytest.mark.parametrize("key", [None, "wrong-key"])
def test_get_endpoints_reject_bad_key(server, path, key):
    assert server.get(path, key=key).status_code == 401


def test_unauthorized_upload_is_not_queued(server, negative_video):
    before = server.get("/health", key=None).json()["jobs"]
    server.post_video(negative_video, key="wrong-key")
    after = server.get("/health", key=None).json()["jobs"]
    assert before == after


# ---------------------------------------------------------------- валидация загрузки

def test_unknown_job_404(server):
    r = server.get("/api/v1/jobs/00000000")
    assert r.status_code == 404


def test_rejects_unsupported_extension(server):
    r = server.post_video(None, filename="notes.txt", content=b"hello")
    assert r.status_code == 415
    assert ".txt" in r.json()["detail"]


def test_rejects_empty_file(server):
    r = server.post_video(None, filename="empty.mov", content=b"")
    assert r.status_code == 400


def test_rejects_oversized_file(server):
    limit_mb = 40  # MAX_UPLOAD_MB тестового сервера (см. conftest)
    r = server.post_video(None, filename="big.mov", content=os.urandom((limit_mb + 1) * 1024 * 1024))
    assert r.status_code == 413
    assert str(limit_mb) in r.json()["detail"]
    assert not list((server.data_dir / "uploads").glob("*")), "отклонённый файл должен быть удалён"


def test_missing_file_field_422(server):
    import requests

    r = requests.post(server.base_url + "/api/v1/analyze", headers=server.headers(), data={"foo": "bar"}, timeout=30)
    assert r.status_code == 422


def test_corrupted_video_results_in_failed(server):
    job_id = server.submit(None, filename="corrupt.mov", content=os.urandom(256 * 1024))
    res = server.wait_job(job_id, timeout=60)
    assert res["status"] == "failed"
    assert "cannot open video" in res["error"]
    assert str(server.data_dir) not in res["error"] and "/uploads/" not in res["error"], "путь сервера не должен уходить клиенту"
    assert "verdict" not in res and "violations" not in res
    # сервис жив и принимает следующие задания
    assert server.get("/health", key=None).json()["status"] == "ok"


# ---------------------------------------------------------------- полный жизненный цикл

def test_analyze_returns_job_id_quickly(server, negative_video):
    t0 = time.time()
    r = server.post_video(negative_video)
    assert r.status_code == 200, r.text
    assert time.time() - t0 < 5, "ПЛАН 8.1: job_id должен возвращаться быстро"
    body = r.json()
    assert JOB_ID_RE.match(body["job_id"]) and body["status"] == "processing"
    server.wait_job(body["job_id"])


def test_processing_status_shape(server):
    job_id = server.submit(video_path(LONGEST_VIDEO))
    r = server.get(f"/api/v1/jobs/{job_id}")
    assert r.status_code == 200
    body = r.json()
    assert body["job_id"] == job_id
    assert body["status"] in ("processing", "done")
    if body["status"] == "processing":
        assert "stage" in body["progress"]
        assert "verdict" not in body and "violations" not in body
    res = server.wait_job(job_id)
    assert_done_result(res)
    assert 140 <= res["meta"]["duration_sec"] <= 145
    assert res["meta"]["processing_time_sec"] < 240, "ПЛАН 8.1: 143-секундное видео обрабатывается < 4 мин"


def _t9_context(res: dict) -> dict:
    ctx = res["meta"].get("context_events") or []
    t9 = [e for e in ctx if e["type_code"] == "T9"]
    assert len(t9) == 1, f"ожидали ровно одно контекстное событие T9, получили {ctx}"
    return t9[0]


def test_lifecycle_x11_t3_violation_with_t9_context(server, positive_video):
    """Х-11: свет гаснет на 16.1 с (разметка — 18 с), экран ноутбука остаётся активным в темноте → нарушение T3.
    T9 — контекстное событие (CONTEXT_ONLY_TYPES): попадает в meta.context_events и не выдаётся как нарушение."""
    job_id = server.submit(positive_video)
    res = server.wait_job(job_id)
    assert_done_result(res)
    assert res["job_id"] == job_id
    assert res["verdict"] == "violation"
    assert [v["type_code"] for v in res["violations"]] == ["T3"], "ровно одно нарушение — T3; T9 остаётся контекстом"
    t3 = res["violations"][0]
    assert 14.0 <= t3["timecode_sec"] <= 20.0, "разметка: 18-я секунда; свет гаснет на 16.1 с"
    assert t3["end_sec"] == pytest.approx(res["meta"]["duration_sec"], abs=0.2), "экран активен до конца ролика"
    assert t3["confidence"] >= 0.7
    assert "экран" in t3["description"] and "освещение" in t3["description"]
    assert "АРМ" in t3["type"]
    assert t3["rule"]["clause"] == "п. 71 пп. 9)" and "V2600038336" in t3["rule"]["url"] and t3["rule"].get("note")
    assert len(t3["evidence_frames"]) == 3
    t9 = _t9_context(res)
    assert 14.0 <= t9["timecode_sec"] <= 20.0
    assert t9["end_sec"] == pytest.approx(res["meta"]["duration_sec"], abs=0.2), "свет не включается до конца ролика"
    assert "яркость" in t9["description"] and "освещение" in t9["name"].lower()
    assert abs(res["meta"]["duration_sec"] - 25.6) < 0.5


def test_x11_verdict_matches_label(server, positive_video):
    """По разметке Х-11 — нарушение (18 с): вердикт violation за счёт T3 (уход в темноте с активным экраном)."""
    res = server.wait_job(server.submit(positive_video))
    assert res["verdict"] == "violation"
    assert any(v["type_code"] == "T3" for v in res["violations"])


def test_lifecycle_clean_video(server, negative_video):
    res = server.wait_job(server.submit(negative_video))
    assert_done_result(res)
    assert res["verdict"] == "no_violation"
    assert res["violations"] == []


def test_result_is_stable_and_repeatable(server, negative_video):
    job_id = server.submit(negative_video)
    a = server.wait_job(job_id)
    b = server.get(f"/api/v1/jobs/{job_id}").json()
    assert a == b, "повторный GET возвращает тот же результат"
    # повторная загрузка того же файла — новое задание с тем же вердиктом
    other = server.wait_job(server.submit(negative_video))
    assert other["job_id"] != job_id and other["verdict"] == a["verdict"]


def test_concurrent_uploads_all_complete(server):
    names = [POSITIVE_VIDEO, NEGATIVE_VIDEO, "Х-22.mov", POSITIVE_VIDEO, NEGATIVE_VIDEO]
    paths = [video_path(n) for n in names]
    with cf.ThreadPoolExecutor(len(paths)) as ex:
        job_ids = list(ex.map(server.submit, paths))
    assert len(set(job_ids)) == len(job_ids)
    results = [server.wait_job(j) for j in job_ids]
    for r in results:
        assert_done_result(r)
        # деградация под нагрузкой допустима, но никогда не молча
        assert isinstance(r["meta"]["degraded"], bool) and isinstance(r["meta"]["analysis_complete"], bool)
        if r["meta"]["degraded"]:
            assert r["meta"]["degraded_reason"], r["meta"]
    assert [r["verdict"] for r in results] == ["violation", "no_violation", "no_violation", "violation", "no_violation"]
    has_t3 = [any(v["type_code"] == "T3" for v in r["violations"]) for r in results]
    has_t9 = [any(e["type_code"] == "T9" for e in r["meta"].get("context_events") or []) for r in results]
    assert has_t3 == has_t9 == [True, False, False, True, False]


def test_uploads_are_deleted_after_processing(server, negative_video):
    server.wait_job(server.submit(negative_video))
    time.sleep(0.5)
    assert not list((server.data_dir / "uploads").glob("*")), "ТЗ: тестовые видео не хранятся после обработки"


def test_health_reports_jobs(server):
    h = server.get("/health", key=None).json()
    assert h["jobs"]["done"] >= 3 and h["jobs"]["failed"] >= 1


# ---------------------------------------------------------------- справочник и пример из ТЗ

def test_rules_endpoint(server):
    body = server.get("/api/v1/rules").json()
    assert set(body["documents"]) == {"R231", "R14", "R53", "R48"}
    assert set(body["rules"]) == {f"T{i}" for i in range(1, 14)}


@pytest.mark.skipif(shutil.which("curl") is None, reason="нет curl")
def test_curl_example_from_tz(server, positive_video):
    """Дословный пример запроса из раздела 10 ТЗ."""
    out = subprocess.run(
        ["curl", "-sS", "-X", "POST", f"{server.base_url}/api/v1/analyze", "-H", f"X-API-Key: {TEST_API_KEY}", "-F", f"file=@{positive_video}"],
        capture_output=True, text=True, timeout=120, check=True,
    ).stdout
    job_id = json.loads(out)["job_id"]
    out = subprocess.run(
        ["curl", "-sS", "-H", f"X-API-Key: {TEST_API_KEY}", f"{server.base_url}/api/v1/jobs/{job_id}"],
        capture_output=True, text=True, timeout=30, check=True,
    ).stdout
    assert json.loads(out)["status"] in ("processing", "done")
    res = server.wait_job(job_id)
    assert_done_result(res)
    assert res["verdict"] == "violation" and res["violations"][0]["type_code"] == "T3"
    _t9_context(res)
