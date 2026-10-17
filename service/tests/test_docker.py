"""Docker e2e: сборка образа, контейнер с пробросом порта (как в docker-compose), полный цикл по API,
и отдельно контейнер без сети (--network none) — проверка «работает без интернета» (ПЛАН 8.1)."""
from __future__ import annotations

import json
import subprocess
import time
import uuid

import pytest
import requests

from .conftest import NEGATIVE_VIDEO, POSITIVE_VIDEO, SERVICE_DIR, TEST_API_KEY, docker_available, free_port, video_path
from .schema import assert_done_result

pytestmark = [pytest.mark.docker, pytest.mark.dataset, pytest.mark.skipif(not docker_available(), reason="Docker-демон недоступен или E2E_SKIP_DOCKER=1")]

IMAGE = "fortebank-cashdesk:cpu"


def _run(*args, timeout=600, check=True, **kw):
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=check, **kw)


@pytest.fixture(scope="module")
def image():
    _run("docker", "compose", "build", cwd=str(SERVICE_DIR), timeout=1200)
    assert _run("docker", "image", "inspect", IMAGE).returncode == 0
    return IMAGE


class Container:
    def __init__(self, name: str):
        self.name = name

    def exec(self, *cmd, timeout=120):
        return _run("docker", "exec", self.name, *cmd, timeout=timeout)

    def logs(self):
        return _run("docker", "logs", self.name, check=False).stdout


def _wait_health(fn, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if fn():
                return
        except Exception:  # noqa: BLE001
            pass
        time.sleep(1)
    raise AssertionError("контейнер не стал здоровым")


@pytest.fixture(scope="module")
def container(image):
    name = f"cashdesk-e2e-{uuid.uuid4().hex[:8]}"
    port = free_port()
    _run("docker", "run", "-d", "--rm", "--name", name, "-p", f"127.0.0.1:{port}:8000", "-e", f"API_KEY={TEST_API_KEY}", image)
    c = Container(name)
    c.base_url = f"http://127.0.0.1:{port}"
    try:
        _wait_health(lambda: requests.get(c.base_url + "/health", timeout=2).status_code == 200)
        yield c
    finally:
        print(c.logs()[-3000:])
        _run("docker", "rm", "-f", name, check=False)


def test_container_api_lifecycle(container):
    h = requests.get(container.base_url + "/health", timeout=5).json()
    assert h["status"] == "ok"
    # 401 без ключа
    with open(video_path(NEGATIVE_VIDEO), "rb") as f:
        assert requests.post(container.base_url + "/api/v1/analyze", files={"file": f}, timeout=60).status_code == 401
    # UI отдаётся
    assert "Контроль кассовой дисциплины" in requests.get(container.base_url + "/", timeout=5).text

    def analyze(name):
        with open(video_path(name), "rb") as f:
            r = requests.post(container.base_url + "/api/v1/analyze", headers={"X-API-Key": TEST_API_KEY}, files={"file": (name, f)}, timeout=120)
        assert r.status_code == 202, r.text
        job_id = r.json()["job_id"]
        deadline = time.time() + 240
        while time.time() < deadline:
            body = requests.get(f"{container.base_url}/api/v1/jobs/{job_id}", headers={"X-API-Key": TEST_API_KEY}, timeout=10).json()
            if body["status"] != "processing":
                return body
            time.sleep(1)
        raise AssertionError("таймаут")

    pos, neg = analyze(POSITIVE_VIDEO), analyze(NEGATIVE_VIDEO)
    assert_done_result(pos)
    assert_done_result(neg)
    assert pos["verdict"] == "violation" and [v["type_code"] for v in pos["violations"]] == ["T3"], pos["violations"]
    assert any(e["type_code"] == "T9" for e in pos["meta"]["context_events"]), pos["meta"]
    assert neg["verdict"] == "no_violation" and not neg["meta"]["context_events"]


def test_container_runs_as_non_root_with_cyrillic_font(container):
    assert container.exec("id", "-u").stdout.strip() != "0"
    assert "DejaVuSans" in container.exec("sh", "-c", "ls /usr/share/fonts/truetype/dejavu").stdout


def test_offline_container(image):
    """Контейнер без сетевого доступа обрабатывает видео: запросы делаем изнутри через docker exec + curl."""
    name = f"cashdesk-offline-{uuid.uuid4().hex[:8]}"
    _run("docker", "run", "-d", "--rm", "--name", name, "--network", "none", "-e", f"API_KEY={TEST_API_KEY}", image)
    c = Container(name)
    try:
        _wait_health(lambda: "ok" in c.exec("curl", "-fsS", "http://127.0.0.1:8000/health").stdout)
        _run("docker", "cp", str(video_path(POSITIVE_VIDEO)), f"{name}:/tmp/video.mov")
        out = c.exec("curl", "-sS", "-X", "POST", "http://127.0.0.1:8000/api/v1/analyze", "-H", f"X-API-Key: {TEST_API_KEY}", "-F", "file=@/tmp/video.mov").stdout
        job_id = json.loads(out)["job_id"]
        deadline = time.time() + 240
        while time.time() < deadline:
            body = json.loads(c.exec("curl", "-sS", "-H", f"X-API-Key: {TEST_API_KEY}", f"http://127.0.0.1:8000/api/v1/jobs/{job_id}").stdout)
            if body["status"] != "processing":
                break
            time.sleep(1)
        assert_done_result(body)
        assert body["verdict"] == "violation" and [v["type_code"] for v in body["violations"]] == ["T3"], body["violations"]
        assert any(e["type_code"] == "T9" for e in body["meta"]["context_events"]), body["meta"]
    finally:
        print(c.logs()[-2000:])
        _run("docker", "rm", "-f", name, check=False)
