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


def test_timeout_still_returns_done_within_limit(fragile_server):
    """Жёсткий дедлайн: задание не «failed» (= необработанное), а done с безопасным вердиктом и пометкой degraded."""
    t0 = time.time()
    res = fragile_server.wait_job(_submit(fragile_server, b"SLOW"), timeout=30)
    elapsed = time.time() - t0
    assert res["status"] == "done", res
    assert res["verdict"] == "no_violation" and res["violations"] == []
    assert res["meta"]["degraded"] is True and any("дедлайн" in r for r in res["meta"]["degrade_reasons"])
    assert elapsed < 4 + 6, f"ответ должен прийти около 4 с, прошло {elapsed:.1f}"
    assert "hard deadline" in fragile_server.log()


def test_worker_restarts_after_timeout(fragile_server):
    res = fragile_server.wait_job(_submit(fragile_server, b"OK"), timeout=30)
    assert res["status"] == "done"
    assert fragile_server.get("/health", key=None).json()["worker_alive"] is True


def test_worker_crash_is_reported_and_recovered(fragile_server):
    res = fragile_server.wait_job(_submit(fragile_server, b"CRASH"), timeout=30)
    assert res["status"] == "done" and res["meta"]["degraded"] is True, res
    assert res["meta"]["fallback"] is True and "аварийно" in res["meta"]["degrade_reasons"][-1]
    res = fragile_server.wait_job(_submit(fragile_server, b"OK"), timeout=30)
    assert res["status"] == "done"


def test_pipeline_exception_gives_safe_answer(fragile_server):
    """Неожиданное исключение пайплайна на валидном видео не должно стоить видео: безопасный ответ + причина."""
    res = fragile_server.wait_job(_submit(fragile_server, b"FAIL"), timeout=30)
    assert res["status"] == "done" and res["verdict"] == "no_violation"
    assert res["meta"]["degraded"] is True and "boom" in " ".join(res["meta"]["degrade_reasons"])


def test_undecodable_file_is_failed_with_reason(fragile_server):
    res = fragile_server.wait_job(_submit(fragile_server, b"DECO"), timeout=30)
    assert res["status"] == "failed" and "cannot open video" in res["error"]


def test_queue_is_sequential_and_survives_mixed_load(fragile_server):
    ids = [_submit(fragile_server, m) for m in (b"OK", b"SLOW", b"OK", b"FAIL", b"OK")]
    results = [fragile_server.wait_job(j, timeout=60) for j in ids]
    assert [r["status"] for r in results] == ["done"] * 5
    degraded = [r["meta"].get("degraded", False) for r in results]
    # SLOW и FAIL — деградированные ответы; поздние OK могут тоже опоздать (часы идут от POST, лимит здесь 4 с)
    assert degraded[0] is False and degraded[1] is True and degraded[3] is True
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


# ---------------------------------------------------------------- деградация, частичные результаты, пул, очистка

def test_hard_deadline_returns_partial_result(pool_server):
    """Пайплайн успел посчитать нарушение и завис (например, на VLM): жёсткий дедлайн отдаёт готовый частичный результат."""
    t0 = time.time()
    res = pool_server.wait_job(_submit(pool_server, b"PART"), timeout=40)
    assert res["status"] == "done" and res["verdict"] == "violation" and len(res["violations"]) == 1
    assert res["violations"][0]["evidence_frames"][0]["image_base64"]
    assert res["meta"]["degraded"] is True
    assert time.time() - t0 < 15 + 8


def test_light_mode_retry_after_crash(pool_server):
    """Процесс упал на видео: повтор в лёгком режиме (правила по яркости) на новом процессе, а не «failed»."""
    res = pool_server.wait_job(_submit(pool_server, b"LCRASH"), timeout=30)
    assert res["status"] == "done", res
    assert res["meta"]["light"] is True


def test_deadlines_are_passed_and_measured_from_post(pool_server):
    res = pool_server.wait_job(_submit(pool_server, b"DEAD"), timeout=30)
    assert res["status"] == "done"
    assert 8.0 < res["meta"]["soft_in"] <= 10.5, res["meta"]      # SOFT = 0.7 * 15 с от POST
    assert 12.0 < res["meta"]["hard_in"] <= 15.0, res["meta"]


def test_queue_wait_is_charged_to_the_job(pool_server):
    """Два воркера заняты по 1.5 с; третье задание ждёт — его дедлайн считается от POST, а не от старта."""
    a, b = _submit(pool_server, b"SPIN"), _submit(pool_server, b"SPIN")
    c = _submit(pool_server, b"DEAD")
    ra, rb, rc = (pool_server.wait_job(j, timeout=30) for j in (a, b, c))
    assert rc["meta"]["soft_in"] < 10.5 - 1.0, rc["meta"]
    assert ra["meta"]["pid"] != rb["meta"]["pid"], "два параллельных задания должны идти в разных процессах"


def test_two_parallel_jobs_finish_together(pool_server):
    t0 = time.time()
    ids = [_submit(pool_server, b"SPIN") for _ in range(2)]
    res = [pool_server.wait_job(j, timeout=30) for j in ids]
    assert all(r["status"] == "done" for r in res)
    assert time.time() - t0 < 1.5 * 2 + 0.5, "2 задания по 1.5 с должны идти параллельно"


def test_five_sequential_jobs(pool_server):
    for _ in range(5):
        r = pool_server.wait_job(_submit(pool_server, b"SPIN"), timeout=30)
        assert r["status"] == "done"


def test_uploads_are_deleted_in_every_outcome(pool_server):
    ids = [_submit(pool_server, m) for m in (b"SPIN", b"FAIL", b"DECO", b"CRASH", b"SLOW")]
    for j in ids:
        pool_server.wait_job(j, timeout=40)
    time.sleep(0.5)
    left = [p.name for p in (pool_server.data_dir / "uploads").glob("*")]
    assert left == [], f"тестовые видео не должны храниться: {left}"


def test_cyrillic_and_odd_filenames_are_accepted(pool_server):
    names = ["Х-11.mov", "видео кассы №3.MOV", "clip.m4v", "no_extension", "C:\\Users\\жюри\\video_017.mov"]
    ids = [pool_server.submit(None, filename=n, content=b"SPIN" + b"\0" * 64) for n in names]
    assert all(pool_server.wait_job(j, timeout=40)["status"] == "done" for j in ids)


def test_unknown_extension_still_rejected(pool_server):
    assert pool_server.post_video(None, filename="notes.txt", content=b"hello").status_code == 415


def test_stale_uploads_are_purged_on_startup(tmp_path):
    from .conftest import start_server, stop_server

    up = tmp_path / "data" / "uploads"
    up.mkdir(parents=True)
    (up / "deadbeef.mov").write_bytes(b"leftover")
    s = start_server(tmp_path, fake_pipeline=True)
    try:
        assert list(up.glob("*")) == []
    finally:
        stop_server(s)


def test_dispatcher_survives_unexpected_exception():
    """Исключение внутри обработки одного задания не убивает диспетчера и не оставляет задание в processing."""
    from app.store import Job, JobStore
    from app.worker import AnalysisWorker

    store = JobStore()
    w = AnalysisWorker(store, workers=1)
    calls = []

    def fake_run(slot, job_id, path):
        calls.append(job_id)
        if job_id == "bad":
            raise RuntimeError("store exploded")
        w._finish_done(job_id, {"verdict": "no_violation", "violations": [], "meta": {}}, time.time())

    w._run_one = fake_run                      # type: ignore[assignment]
    w._ensure_child = lambda slot, deadline: True   # type: ignore[assignment]
    w._kill_child = lambda slot: None          # type: ignore[assignment]
    for jid in ("bad", "good"):
        store.add(Job(job_id=jid, filename="x.mov", path="/nonexistent/x.mov"))
        w.submit(jid, "/nonexistent/x.mov")
    w.start()
    deadline = time.time() + 10
    while time.time() < deadline and any(store.get(j).status == "processing" for j in ("bad", "good")):
        time.sleep(0.05)
    w.stop()
    assert calls == ["bad", "good"]
    assert store.get("bad").status == "done" and store.get("bad").result["meta"]["degraded"] is True
    assert store.get("good").status == "done"
