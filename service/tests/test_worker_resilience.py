"""Устойчивость очереди: таймаут, падение процесса, исключение в пайплайне, восстановление воркера.
Используется подменный пайплайн (tests/fixtures/fake_pipeline) и JOB_TIMEOUT_SEC=4."""
from __future__ import annotations

import time


def _submit(server, marker: bytes) -> str:
    return server.submit(None, filename="probe.mov", content=marker + b"\0" * 1024)


def test_progress_is_reported_while_processing(fragile_server):
    job_id = _submit(fragile_server, b"OK")
    seen = set()
    deadline = time.time() + 10
    while time.time() < deadline:
        body = fragile_server.get(f"/api/v1/jobs/{job_id}").json()
        if body["status"] != "processing":
            break
        seen.add(body["progress"].get("stage"))
        time.sleep(0.1)
    assert body["status"] == "done", body
    assert "fake-stage" in seen, f"ожидали стадию из пайплайна, видели {seen}"
    assert body["meta"]["pipeline_version"] == "fake-test"


def test_timeout_marks_job_failed_within_limit(fragile_server):
    t0 = time.time()
    res = fragile_server.wait_job(_submit(fragile_server, b"SLOW"), timeout=30)
    elapsed = time.time() - t0
    assert res["status"] == "failed"
    assert "timeout" in res["error"] and "4 s" in res["error"]
    assert elapsed < 4 + 6, f"таймаут должен срабатывать около 4 с, прошло {elapsed:.1f}"
    assert "timed out" in fragile_server.log()


def test_worker_restarts_after_timeout(fragile_server):
    res = fragile_server.wait_job(_submit(fragile_server, b"OK"), timeout=30)
    assert res["status"] == "done"
    assert fragile_server.get("/health", key=None).json()["worker_alive"] is True


def test_worker_crash_is_reported_and_recovered(fragile_server):
    res = fragile_server.wait_job(_submit(fragile_server, b"CRASH"), timeout=30)
    assert res["status"] == "failed" and "crashed" in res["error"]
    res = fragile_server.wait_job(_submit(fragile_server, b"OK"), timeout=30)
    assert res["status"] == "done"


def test_pipeline_exception_is_reported(fragile_server):
    res = fragile_server.wait_job(_submit(fragile_server, b"FAIL"), timeout=30)
    assert res["status"] == "failed"
    assert res["error"] == "RuntimeError: boom"
    assert "verdict" not in res


def test_queue_is_sequential_and_survives_mixed_load(fragile_server):
    ids = [_submit(fragile_server, m) for m in (b"OK", b"SLOW", b"OK", b"FAIL", b"OK")]
    results = [fragile_server.wait_job(j, timeout=60) for j in ids]
    assert [r["status"] for r in results] == ["done", "failed", "done", "failed", "done"]
    h = fragile_server.get("/health", key=None).json()
    assert h["queue"] == 0 and h["jobs"]["processing"] == 0


def test_violation_passes_through_api_unchanged(fragile_server):
    """Нарушение из пайплайна доходит до клиента ровно в схеме ТЗ (type, timecode_sec, description, evidence_frames)."""
    from .schema import assert_done_result, assert_jpeg_base64

    res = fragile_server.wait_job(_submit(fragile_server, b"VIOL"), timeout=30)
    assert_done_result(res)
    assert res["verdict"] == "violation" and len(res["violations"]) == 1
    v = res["violations"][0]
    assert v["type"].startswith("Тестовое нарушение") and v["timecode_sec"] == 1.5 and v["end_sec"] == 4.0
    assert [f["timestamp_sec"] for f in v["evidence_frames"]] == [1.5, 2.5]
    img = assert_jpeg_base64(v["evidence_frames"][0]["image_base64"])
    assert img.shape[:2] == (64, 48)
    assert v["rule"] == {"document": "Правила № 231 (тест)", "clause": "п. 20", "url": "https://adilet.zan.kz/rus/docs/V1900019680"}
    # meta (в т.ч. дорожки состояний для шкалы времени UI) проходит через API без изменений
    tl = res["meta"]["timeline"]
    assert tl["duration"] == 6.0 and tl["lanes"]["visitors"] == [[3.0, 6.0, 1]] and tl["lanes"]["cashier"] == [[0.0, 1.5]]
