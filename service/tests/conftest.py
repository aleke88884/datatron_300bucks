"""Общие фикстуры e2e-тестов: запуск настоящего сервера (uvicorn в подпроцессе), клиент, датасет."""
from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pytest
import requests

SERVICE_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = SERVICE_DIR.parent
VIDEOS_DIR = ROOT_DIR / "open" / "videos"
LABELS_XLSX = ROOT_DIR / "open" / "open_labels.xlsx"
FAKE_PIPELINE_DIR = SERVICE_DIR / "tests" / "fixtures" / "fake_pipeline"
TEST_API_KEY = "e2e-test-key-0123456789abcdef"

RESULTS_DIR = SERVICE_DIR / "results"
# Датасет-тесты берут детекции/состояния из results/ (FEATURES_CACHE), если они есть: без детектора прогон 42 видео
# занимает ~15 с вместо ~5 мин. Живой путь (детектор YOLOX по видео) проверяется отдельным тестом на одном видео.
if (RESULTS_DIR / "perception").exists() and "FEATURES_CACHE" not in os.environ:
    os.environ["FEATURES_CACHE"] = str(RESULTS_DIR)

POSITIVE_VIDEO = "Х-11.mov"   # нарушение T3 (экран АРМ активен после выключения света; контекст T9), разметка 18 с
NEGATIVE_VIDEO = "Х-3.mov"    # без нарушения
LONGEST_VIDEO = "Х-14.mov"    # 143 с — самое длинное видео датасета


def video_path(name: str) -> Path:
    p = VIDEOS_DIR / name
    if not p.exists():
        pytest.skip(f"нет видео {p}")
    return p


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@dataclass
class Server:
    base_url: str
    api_key: str
    data_dir: Path
    proc: subprocess.Popen
    log_path: Path

    # --- клиент ---------------------------------------------------------
    def headers(self, key: Optional[str] = TEST_API_KEY) -> dict:
        return {"X-API-Key": key} if key is not None else {}

    def get(self, path: str, key: Optional[str] = TEST_API_KEY, **kw) -> requests.Response:
        return requests.get(self.base_url + path, headers=self.headers(key), timeout=30, **kw)

    def post_video(self, path: Path | str, key: Optional[str] = TEST_API_KEY, filename: Optional[str] = None, content: Optional[bytes] = None) -> requests.Response:
        if content is not None:
            files = {"file": (filename or "video.mov", content, "video/quicktime")}
            return requests.post(self.base_url + "/api/v1/analyze", headers=self.headers(key), files=files, timeout=120)
        with open(path, "rb") as f:
            files = {"file": (filename or Path(path).name, f, "video/quicktime")}
            return requests.post(self.base_url + "/api/v1/analyze", headers=self.headers(key), files=files, timeout=120)

    def submit(self, path: Path | str, **kw) -> str:
        r = self.post_video(path, **kw)
        assert r.status_code == 200, r.text
        return r.json()["job_id"]

    def wait_job(self, job_id: str, timeout: float = 240.0, poll: float = 0.5) -> dict:
        deadline = time.time() + timeout
        last = None
        while time.time() < deadline:
            r = self.get(f"/api/v1/jobs/{job_id}")
            assert r.status_code == 200, r.text
            last = r.json()
            if last["status"] != "processing":
                return last
            time.sleep(poll)
        raise AssertionError(f"job {job_id} не завершился за {timeout} с: {last}")

    def log(self) -> str:
        return self.log_path.read_text(errors="replace")


def start_server(tmp_dir: Path, env_overrides: Optional[dict] = None, fake_pipeline: bool = False) -> Server:
    port = free_port()
    data_dir = tmp_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    # Сервер e2e работает как в проде — без кеша признаков: загрузки называются по job_id, кеш для них
    # всё равно не находится, а запись results/<feature>/<job_id>.json засоряла бы рабочее дерево.
    env.pop("FEATURES_CACHE", None)
    env.update(
        {
            "API_KEY": TEST_API_KEY,
            "DATA_DIR": str(data_dir),
            "LOG_LEVEL": "INFO",
            "MAX_UPLOAD_MB": "40",
            "JOB_TIMEOUT_SEC": "240",
            "KEEP_UPLOADS": "false",
        }
    )
    env.update(env_overrides or {})
    # cwd — временная папка: тогда sys.path[0] не содержит service/, и подменный pipeline из PYTHONPATH
    # имеет приоритет над настоящим (важно для тестов устойчивости воркера).
    pythonpath = [str(SERVICE_DIR)]
    if fake_pipeline:
        pythonpath.insert(0, str(FAKE_PIPELINE_DIR))
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    log_path = tmp_dir / "server.log"
    log_f = open(log_path, "w")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(port), "--log-level", "info"],
        cwd=str(tmp_dir),
        env=env,
        stdout=log_f,
        stderr=subprocess.STDOUT,
    )
    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + 60
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"сервер упал при старте:\n{log_path.read_text(errors='replace')}")
        try:
            if requests.get(base + "/health", timeout=2).status_code == 200:
                return Server(base_url=base, api_key=TEST_API_KEY, data_dir=data_dir, proc=proc, log_path=log_path)
        except requests.RequestException:
            pass
        time.sleep(0.25)
    proc.kill()
    raise RuntimeError(f"сервер не поднялся за 60 с:\n{log_path.read_text(errors='replace')}")


def stop_server(server: Server) -> None:
    if server.proc.poll() is None:
        server.proc.terminate()
        try:
            server.proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.proc.kill()
            server.proc.wait(timeout=5)


@pytest.fixture(scope="session")
def server(tmp_path_factory) -> Server:
    """Настоящий сервер с настоящим пайплайном (один на всю сессию тестов)."""
    s = start_server(tmp_path_factory.mktemp("server"))
    yield s
    stop_server(s)


@pytest.fixture(scope="module")
def fragile_server(tmp_path_factory) -> Server:
    """Сервер с подменным пайплайном и коротким таймаутом — для проверки таймаута/падений воркера."""
    s = start_server(tmp_path_factory.mktemp("fragile"), env_overrides={"JOB_TIMEOUT_SEC": "4", "WORKERS": "2"}, fake_pipeline=True)
    yield s
    stop_server(s)


@pytest.fixture(scope="module")
def pool_server(tmp_path_factory) -> Server:
    """Подменный пайплайн, 2 воркера, лимит 15 с (мягкий дедлайн 10.5 с): деградация, частичные результаты, параллелизм."""
    s = start_server(tmp_path_factory.mktemp("pool"), env_overrides={"JOB_TIMEOUT_SEC": "15", "WORKERS": "2"}, fake_pipeline=True)
    yield s
    stop_server(s)


@pytest.fixture(scope="session")
def positive_video() -> Path:
    return video_path(POSITIVE_VIDEO)


@pytest.fixture(scope="session")
def negative_video() -> Path:
    return video_path(NEGATIVE_VIDEO)


def docker_available() -> bool:
    if os.environ.get("E2E_SKIP_DOCKER") == "1" or shutil.which("docker") is None:
        return False
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=20).returncode == 0
    except (subprocess.SubprocessError, OSError):
        return False
