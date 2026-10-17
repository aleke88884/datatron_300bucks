"""VLM-слой: локальная мультимодальная модель уточняет описания событий по кадрам (ПЛАН 5.1, п. 4).

Зачем. Правила T1–T13 дают точный таймкод и шаблонное описание («кассир вне зоны стола, на столе деньги»).
Модель по 1–3 кадрам события переписывает шаблон в 2–3 предложения «что видно на кадрах» — человеческое
объяснение для жюри и проверяющего. Метод `ask_yes_no` отвечает «да/нет» по фрагменту кадра — его использует
страховочная сетка от неизвестных типов нарушений (pipeline/safety_net.py; ПЛАН 5.4: ответ модели сам по себе
событие НЕ добавляет — нужны детектор, повтор на нескольких кадрах и отрицательный контроль). `checklist` —
прежний пакетный вариант вопросов по кадрам.

Модель. Qwen/Qwen2.5-VL-7B-Instruct, лицензия Apache-2.0 (коммерческое использование разрешено). Работает
полностью локально через transformers; облачные API не используются (условие кейса). Qwen2.5-VL-3B НЕ
использовать: у него Qwen Research License (только некоммерческое). Веса: ~16 ГБ в bf16; на GPU с 8–12 ГБ
памяти — 4-битная загрузка через bitsandbytes (VLM_4BIT=1, ~6 ГБ).

Как включить (см. docs_vlm_section.md):
    pip install -r requirements-vlm.txt
    python scripts/get_vlm.py                 # веса в service/models/vlm/
    VLM_ENABLED=1 в .env (и при необходимости VLM_4BIT=1)

Переменные окружения:
    VLM_ENABLED       1/true — включить слой (по умолчанию выключен)
    VLM_MODEL_PATH    папка с весами (по умолчанию service/models/vlm)
    VLM_4BIT          1 — загрузка в 4 бит через bitsandbytes (только CUDA)
    VLM_TIMEOUT_SEC   таймаут одной генерации, с (по умолчанию 20)
    VLM_MAX_EVENTS    не более N событий на видео (по умолчанию 5)
    VLM_BUDGET_SEC    суммарный бюджет времени VLM на видео, с (по умолчанию 60)
    VLM_MAX_SIDE      кадр уменьшается до этой длинной стороны, px (по умолчанию 768)

Бэкенд llama.cpp (VLM_BACKEND=llamacpp) — для CPU/Apple Silicon без torch. Локальный `llama-server`
(llama.cpp, MIT) с GGUF-весами Qwen3-VL-2B-Instruct (Apache-2.0, ~1.5 ГБ: Q4_K_M + mmproj Q8_0); общение по
HTTP на 127.0.0.1, наружу ничего не уходит. Сервер либо уже запущен (VLM_SERVER_URL), либо модуль сам запускает
его из VLM_LLAMA_BIN (по умолчанию `llama-server` из PATH) с весами из VLM_GGUF_DIR (service/models/vlm_gguf).
Скачать веса: python scripts/get_vlm.py --gguf. Замерено на MacBook Air M3 8 ГБ (Metal): ~1.3 с на новый кадр
448 px, ~0.3 с на следующий вопрос по тому же кадру.

    VLM_BACKEND       transformers (по умолчанию) | llamacpp
    VLM_SERVER_URL    адрес llama-server (по умолчанию http://127.0.0.1:8977)
    VLM_GGUF_DIR      папка с *.gguf и mmproj-*.gguf (по умолчанию service/models/vlm_gguf)
    VLM_LLAMA_BIN     путь к llama-server (по умолчанию из PATH)
    VLM_DESCRIBE      1 — с бэкендом llama.cpp также переписывать описания событий (по умолчанию нет: 2B-модель
                      используется только для «да/нет» страховочной сетки)
    VLM_PROBE_MAX_SEC замер скорости при первом использовании: если один ответ «да/нет» по кадру дольше этого
                      (по умолчанию 8 с), страховочная сетка (pipeline/safety_net.py) модель не использует
    VLM_MIN_RAM_GB    transformers без CUDA: минимальный объём ОЗУ (по умолчанию 32 ГБ — 7B в float32 ≈ 30 ГБ);
                      меньше — загрузка не начинается (иначе своп и срыв лимита 5 минут)

Без модели. Если слой выключен, весов нет или torch/transformers не установлены, модуль всё равно импортируется,
`Describer.available()` возвращает False, `describe_violations` ничего не меняет — остаются шаблонные описания
из правил. Любая ошибка модели ловится и логируется; пайплайн никогда не падает из-за VLM.

Что модель НЕ делает: не ставит вердикт, не меняет тип, таймкод и уверенность события, не делает юридических
выводов (слова «похитил/украл/мошенничество» отбраковываются постобработкой), не описывает внешность людей
(только роли: кассир/клиент/посторонний). Нерелевантный или пустой ответ → None → остаётся шаблон.
"""
from __future__ import annotations

import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

log = logging.getLogger("pipeline.vlm")

SERVICE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = SERVICE_DIR / "models" / "vlm"
DEFAULT_REPO = "Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_LICENSE = "Apache-2.0"

MAX_FRAMES = 3            # ТЗ: 1–3 кадра на событие
MAX_NEW_TOKENS = 120      # 2–3 предложения на русском
MIN_DESCRIPTION_CHARS = 20   # как в проверке схемы ответа (tests/schema.py)
MAX_DESCRIPTION_CHARS = 600
MAX_SENTENCES = 3

# Юридические выводы и оценочные слова: такой ответ отбраковывается целиком (ПЛАН 4.4 — только наблюдаемое).
FORBIDDEN_RE = re.compile(
    r"похит|украл|украд|крад[её]т|кража|мошенн|воров|вор\b|виновн|виновен|преступ|хищени|присво|злоумышл",
    re.IGNORECASE,
)
# Описания внешности людей — запрещены (не идентифицируем лиц).
APPEARANCE_RE = re.compile(
    r"\b(мужчин|женщин|девушк|парень|парн[яе]|блондин|брюнет|лысы|бород|очках|\bлицо\b|национальн)",
    re.IGNORECASE,
)
CJK_RE = re.compile(r"[一-鿿぀-ヿ]")
CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")
LATIN_RE = re.compile(r"[A-Za-z]")

SYSTEM_PROMPT = (
    "Ты помощник службы внутреннего контроля банка. Ты смотришь кадры с камеры видеонаблюдения в кассовом "
    "помещении и описываешь ТОЛЬКО то, что на них видно: нейтрально, кратко, по-русски. Ты не делаешь выводов "
    "о вине, умысле или нарушении закона и не описываешь внешность людей."
)


def _env_flag(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def model_dir() -> Path:
    p = os.environ.get("VLM_MODEL_PATH", "").strip()
    return Path(p).expanduser() if p else DEFAULT_MODEL_DIR


def weights_present(path: Optional[Path] = None) -> bool:
    """Папка с весами считается готовой, если есть config.json и хотя бы один *.safetensors (или *.bin)."""
    d = path or model_dir()
    if not d.is_dir() or not (d / "config.json").is_file():
        return False
    return any(d.glob("*.safetensors")) or any(d.glob("*.bin"))


# ----------------------------------------------------------------------------- бэкенд llama.cpp

DEFAULT_GGUF_DIR = SERVICE_DIR / "models" / "vlm_gguf"
DEFAULT_SERVER_URL = "http://127.0.0.1:8977"
GGUF_REPO = "Qwen/Qwen3-VL-2B-Instruct-GGUF"   # Apache-2.0
GGUF_FILES = ("Qwen3VL-2B-Instruct-Q4_K_M.gguf", "mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf")


def backend() -> str:
    b = (os.environ.get("VLM_BACKEND") or "transformers").strip().lower()
    return "llamacpp" if b in {"llamacpp", "llama.cpp", "llama", "gguf"} else "transformers"


def gguf_dir() -> Path:
    p = os.environ.get("VLM_GGUF_DIR", "").strip()
    return Path(p).expanduser() if p else DEFAULT_GGUF_DIR


def gguf_files(d: Optional[Path] = None) -> Optional[tuple[Path, Path]]:
    """(модель, mmproj) в папке GGUF или None."""
    d = d or gguf_dir()
    if not d.is_dir():
        return None
    mm = sorted(p for p in d.glob("*.gguf") if p.name.lower().startswith("mmproj"))
    lm = sorted(p for p in d.glob("*.gguf") if not p.name.lower().startswith("mmproj"))
    return (lm[0], mm[0]) if lm and mm else None


def llama_bin() -> Optional[str]:
    import shutil

    p = os.environ.get("VLM_LLAMA_BIN", "").strip()
    if p:
        return p if Path(p).expanduser().is_file() else None
    for cand in (gguf_dir() / "bin" / "llama-server", gguf_dir() / "bin" / "llama-server.exe"):
        if cand.is_file():
            return str(cand)
    return shutil.which("llama-server")


def server_url() -> str:
    return (os.environ.get("VLM_SERVER_URL") or DEFAULT_SERVER_URL).rstrip("/")


def _http_json(url: str, body: Optional[dict] = None, timeout: float = 5.0) -> Any:
    import json
    import urllib.request

    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 — только локальный адрес сервера
        return json.loads(r.read().decode("utf-8") or "null")


def server_healthy(url: Optional[str] = None, timeout: float = 1.5) -> bool:
    try:
        d = _http_json(f"{url or server_url()}/health", timeout=timeout)
        return isinstance(d, dict) and d.get("status") == "ok"
    except Exception:  # noqa: BLE001
        return False


def total_ram_gb() -> Optional[float]:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError, AttributeError):
        return None


def _image_to_data_url(img: Any) -> str:
    """PIL.Image или BGR np.ndarray → data:image/jpeg;base64."""
    import base64
    import io

    if isinstance(img, np.ndarray):
        import cv2

        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        raw = buf.tobytes() if ok else b""
    else:
        bio = io.BytesIO()
        img.convert("RGB").save(bio, format="JPEG", quality=88)
        raw = bio.getvalue()
    return "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")


# ----------------------------------------------------------------------------- постобработка ответа


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?…])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def sanitize_description(text: Optional[str], template: str = "") -> Optional[str]:
    """Проверка и чистка ответа модели. None — ответ отбракован, остаётся шаблон.

    Отбраковывается: пустое/слишком короткое, иероглифы, не русский язык, юридические выводы (FORBIDDEN_RE),
    описание внешности (APPEARANCE_RE), дословный повтор шаблона. Остаётся не более MAX_SENTENCES предложений.
    """
    if not text:
        return None
    t = str(text).strip()
    # убираем markdown-обёртки, кавычки по краям, служебные префиксы
    t = re.sub(r"^```[a-z]*\s*|\s*```$", "", t, flags=re.IGNORECASE).strip()
    t = re.sub(r"^(описание|ответ)\s*[:：]\s*", "", t, flags=re.IGNORECASE).strip()
    t = t.strip("«»\"'“” \n\t")
    t = re.sub(r"\s+", " ", t)
    if len(t) < MIN_DESCRIPTION_CHARS:
        return None
    if CJK_RE.search(t):
        return None
    cyr, lat = len(CYRILLIC_RE.findall(t)), len(LATIN_RE.findall(t))
    if cyr == 0 or lat > cyr:
        return None
    if FORBIDDEN_RE.search(t) or APPEARANCE_RE.search(t):
        log.info("vlm: ответ отбракован постобработкой: %r", t[:120])
        return None
    sentences = _split_sentences(t)[:MAX_SENTENCES]
    out = " ".join(sentences)
    if out and out[-1] not in ".!?…":
        out += "."
    if len(out) > MAX_DESCRIPTION_CHARS:
        out = out[:MAX_DESCRIPTION_CHARS].rsplit(" ", 1)[0].rstrip(",;:") + "…"
    if len(out) < MIN_DESCRIPTION_CHARS:
        return None
    if template and out.strip().lower() == template.strip().lower():
        return None  # модель просто повторила шаблон — смысла подменять нет
    return out


def parse_yes_no(text: Optional[str]) -> Optional[bool]:
    """«Да»/«Нет» (и yes/no) в начале ответа → True/False; всё остальное → None (неуверенно)."""
    if not text:
        return None
    t = str(text).strip().lower().lstrip("«\"'*- ")
    m = re.match(r"^(да|нет|yes|no)\b", t)
    if not m:
        return None
    return m.group(1) in ("да", "yes")


# ----------------------------------------------------------------------------- подготовка кадров


def _prepare_images(frames: list[np.ndarray], max_side: Optional[int] = None) -> list[Any]:
    """BGR np.ndarray → PIL.Image RGB, длинная сторона ≤ VLM_MAX_SIDE (меньше визуальных токенов — быстрее)."""
    from PIL import Image

    max_side = max_side or _env_int("VLM_MAX_SIDE", 768)
    images = []
    for fr in frames[:MAX_FRAMES]:
        if fr is None or not isinstance(fr, np.ndarray) or fr.ndim != 3:
            continue
        rgb = np.ascontiguousarray(fr[:, :, ::-1])
        img = Image.fromarray(rgb)
        w, h = img.size
        scale = max_side / float(max(w, h))
        if scale < 1.0:
            img = img.resize((max(28, int(w * scale)), max(28, int(h * scale))), Image.BILINEAR)
        images.append(img)
    return images


# ----------------------------------------------------------------------------- модель


class Describer:
    """Ленивая обёртка над Qwen2.5-VL. Экземпляр один на процесс (см. get_describer())."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path else None
        self.model = None
        self.processor = None
        self.device = "cpu"
        self._lock = threading.Lock()
        self._load_error: Optional[str] = None
        self.backend = backend()
        self._server: Optional[str] = None      # llama.cpp: адрес готового сервера
        self._proc = None                       # llama.cpp: процесс, запущенный нами
        self.latency_sec: Optional[float] = None
        self.model_name = DEFAULT_REPO if self.backend == "transformers" else GGUF_REPO

    # --- доступность

    @property
    def model_path(self) -> Path:
        return self.path or model_dir()

    def available(self) -> bool:
        """Слой включён (VLM_ENABLED=1) и веса на месте; ошибка предыдущей загрузки → False (не пробуем снова).
        llama.cpp: сервер уже отвечает по VLM_SERVER_URL или есть GGUF-веса и бинарник llama-server."""
        if not _env_flag("VLM_ENABLED", False):
            return False
        if self._load_error:
            return False
        if self.backend == "llamacpp":
            if self._server or server_healthy():
                return True
            return gguf_files() is not None and llama_bin() is not None
        return weights_present(self.model_path)

    def unavailable_reason(self) -> str:
        if not _env_flag("VLM_ENABLED", False):
            return "VLM_ENABLED is off"
        if self._load_error:
            return f"load error: {self._load_error}"
        if self.backend == "llamacpp":
            if gguf_files() is None:
                return f"no GGUF weights in {gguf_dir()} (python scripts/get_vlm.py --gguf)"
            if llama_bin() is None:
                return "llama-server binary not found (VLM_LLAMA_BIN)"
            return ""
        if not weights_present(self.model_path):
            return f"no weights in {self.model_path}"
        return ""

    @property
    def loaded(self) -> bool:
        if self.backend == "llamacpp":
            return self._server is not None
        return self.model is not None and self.processor is not None

    # --- загрузка

    def load(self) -> bool:
        """Загружает модель и процессор (один раз). torch/transformers импортируются только здесь.

        Возвращает True, если модель готова. При любой ошибке (нет пакетов, не хватает памяти, битые веса)
        пишет в лог, запоминает ошибку и возвращает False — дальше слой считается недоступным.
        """
        if self.loaded:
            return True
        if self._load_error:
            return False
        with self._lock:
            if self.loaded:
                return True
            if self._load_error:
                return False
            t0 = time.time()
            try:
                if self.backend == "llamacpp":
                    self._load_llamacpp()
                else:
                    self._load_unlocked()
            except Exception as exc:  # noqa: BLE001 — слой необязательный, пайплайн не должен падать
                self._load_error = f"{type(exc).__name__}: {exc}"
                self.model = self.processor = None
                log.exception("vlm: не удалось загрузить модель из %s: %s", self.model_path, exc)
                return False
            log.info("vlm: модель %s загружена за %.1f с (device=%s)", self.model_path, time.time() - t0, self.device)
            return True

    def _load_unlocked(self) -> None:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        path = str(self.model_path)
        if not weights_present(self.model_path):
            raise FileNotFoundError(f"веса VLM не найдены в {path} (python scripts/get_vlm.py)")
        cuda = torch.cuda.is_available()
        if not cuda:
            # 7B в float32 на CPU ≈ 30 ГБ ОЗУ: на обычном ноутбуке это своп и срыв лимита 5 минут — не начинаем.
            # Для CPU/Apple Silicon есть бэкенд llama.cpp (VLM_BACKEND=llamacpp, 2B GGUF ≈ 1.5 ГБ).
            ram = total_ram_gb()
            need = _env_float("VLM_MIN_RAM_GB", 32.0)
            if ram is not None and ram < need:
                raise MemoryError(f"CPU-загрузка {DEFAULT_REPO} требует ≥ {need:.0f} ГБ ОЗУ, есть {ram:.1f} ГБ; "
                                  f"используйте VLM_BACKEND=llamacpp")
        dtype = torch.bfloat16 if cuda else torch.float32
        kwargs: dict[str, Any] = {"device_map": "auto" if cuda else "cpu"}
        if cuda and _env_flag("VLM_4BIT", False):
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                load_in_4bit=True,
            )
        try:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(path, dtype=dtype, **kwargs)
        except TypeError:  # transformers < 4.56: параметр назывался torch_dtype
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(path, torch_dtype=dtype, **kwargs)
        model.eval()
        # min/max_pixels ограничивают число визуальных токенов на кадр (28×28 px на токен)
        processor = AutoProcessor.from_pretrained(path, min_pixels=128 * 28 * 28, max_pixels=640 * 28 * 28)
        self.model, self.processor = model, processor
        self.device = "cuda" if cuda else "cpu"

    def _load_llamacpp(self) -> None:
        """Подключение к llama-server: уже запущенный по VLM_SERVER_URL или запуск своего процесса."""
        url = server_url()
        if server_healthy(url):
            self._server = url
            return
        files = gguf_files()
        binary = llama_bin()
        if files is None or binary is None:
            raise FileNotFoundError(self.unavailable_reason() or "llama.cpp backend is not configured")
        import subprocess
        from urllib.parse import urlparse

        u = urlparse(url)
        host, port = u.hostname or "127.0.0.1", u.port or 8977
        if host not in {"127.0.0.1", "localhost"}:
            raise RuntimeError(f"сервер {url} не отвечает, а запускать свой можно только на localhost")
        cmd = [binary, "-m", str(files[0]), "--mmproj", str(files[1]), "--host", host, "--port", str(port),
               "-c", str(_env_int("VLM_CTX", 4096)), "--no-warmup"]
        log.info("vlm: запуск %s", " ".join(cmd))
        self._proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # noqa: S603
        import atexit

        atexit.register(self.close)   # сервер, запущенный воркером, останавливается вместе с ним
        deadline = time.time() + _env_float("VLM_SERVER_START_SEC", 90.0)
        while time.time() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(f"llama-server завершился с кодом {self._proc.returncode}")
            if server_healthy(url):
                self._server = url
                return
            time.sleep(0.5)
        self.close()
        raise TimeoutError("llama-server не поднялся за отведённое время")

    def close(self) -> None:
        """Останавливает llama-server, если его запустили мы."""
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=5)
            except Exception:  # noqa: BLE001
                pass
            self._proc = None
            self._server = None

    # --- генерация

    def _generate_llamacpp(self, images: list[Any], user_text: str, max_new_tokens: int, timeout_sec: float,
                           system: Optional[str] = SYSTEM_PROMPT) -> Optional[str]:
        """Запрос к llama-server (OpenAI-совместимый /v1/chat/completions). None — таймаут или ошибка."""
        content: list[dict[str, Any]] = [{"type": "image_url", "image_url": {"url": _image_to_data_url(im)}} for im in images]
        content.append({"type": "text", "text": user_text})
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": content})
        body = {"messages": messages, "max_tokens": int(max_new_tokens), "temperature": 0, "seed": 0}
        t0 = time.time()
        try:
            d = _http_json(f"{self._server}/v1/chat/completions", body, timeout=timeout_sec)
            text = d["choices"][0]["message"]["content"]
        except Exception as exc:  # noqa: BLE001
            log.warning("vlm: llama.cpp запрос не удался (%.1f с): %s", time.time() - t0, exc)
            return None
        log.debug("vlm: llama.cpp %.1f с: %r", time.time() - t0, (text or "")[:160])
        return text

    def _generate(self, images: list[Any], user_text: str, max_new_tokens: int, timeout_sec: float) -> Optional[str]:
        if self.backend == "llamacpp":
            if not self.loaded:
                return None
            return self._generate_llamacpp(images, user_text, max_new_tokens, timeout_sec)
        return self._generate_transformers(images, user_text, max_new_tokens, timeout_sec)

    def ask_yes_no(self, image: np.ndarray, question: str) -> Optional[bool]:
        """Один вопрос «да/нет» по одному изображению (BGR). None — нет ответа или ответ не распознан.
        Используется страховочной сеткой (pipeline/safety_net.py); вопрос — на английском (так надёжнее у малых
        моделей), ответ модели наружу не попадает — только True/False/None."""
        if not self.available() or not self.load():
            return None
        try:
            images = _prepare_images([image], max_side=_env_int("VLM_CHECK_SIDE", 448))
            if not images:
                return None
            timeout = _env_float("VLM_TIMEOUT_SEC", 20.0)
            if self.backend == "llamacpp":
                raw = self._generate_llamacpp(images, question, 4, timeout, system=None)
            else:
                raw = self._generate(images, question, 4, timeout)
            return parse_yes_no(raw)
        except Exception as exc:  # noqa: BLE001
            log.exception("vlm: ask_yes_no failed: %s", exc)
            return None

    def fast_enough(self) -> bool:
        """Замер скорости на этой машине (один раз): ответ «да/нет» по синтетическому кадру 448 px.
        Дольше VLM_PROBE_MAX_SEC (8 с) — модель для страховочной сетки не используется (лимит 5 минут на видео)."""
        if not self.loaded:
            return False
        if self.latency_sec is None:
            img = np.full((448, 448, 3), 200, np.uint8)
            img[150:300, 150:300] = (40, 90, 160)
            t0 = time.time()
            ans = self.ask_yes_no(img, "Is there a square in this picture? Answer yes or no.")
            self.latency_sec = round(time.time() - t0, 2)
            log.info("vlm: замер скорости %.2f с (ответ %s)", self.latency_sec, ans)
            if ans is None:
                self.latency_sec = float("inf")
        return self.latency_sec <= _env_float("VLM_PROBE_MAX_SEC", 8.0)

    def _generate_transformers(self, images: list[Any], user_text: str, max_new_tokens: int, timeout_sec: float) -> Optional[str]:
        """Один запрос к модели с таймаутом. None — таймаут или ошибка.

        Генерация идёт в отдельном потоке; основной поток ждёт не дольше timeout_sec. Дополнительно в generate
        передаётся StoppingCriteria с дедлайном, чтобы «опоздавший» поток остановился на следующем токене
        и освободил GPU, а не продолжал считать в фоне.
        """
        if not self.loaded:
            return None
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        deadline = time.monotonic() + timeout_sec

        class _Deadline(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):  # noqa: ANN001
                return time.monotonic() > deadline

        content: list[dict[str, Any]] = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": user_text})
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": content},
        ]
        result: dict[str, Any] = {}

        def _run() -> None:
            try:
                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = self.processor(text=[text], images=images or None, padding=True, return_tensors="pt")
                inputs = inputs.to(self.model.device)
                with torch.inference_mode():
                    out = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        stopping_criteria=StoppingCriteriaList([_Deadline()]),
                    )
                trimmed = out[:, inputs["input_ids"].shape[1]:]
                result["text"] = self.processor.batch_decode(
                    trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
            except Exception as exc:  # noqa: BLE001
                result["error"] = f"{type(exc).__name__}: {exc}"

        th = threading.Thread(target=_run, name="vlm-generate", daemon=True)
        t0 = time.time()
        th.start()
        th.join(timeout_sec)
        if th.is_alive():
            log.warning("vlm: таймаут генерации %.0f с — ответ отброшен", timeout_sec)
            return None
        if "error" in result:
            log.warning("vlm: ошибка генерации: %s", result["error"])
            return None
        log.debug("vlm: генерация %.1f с: %r", time.time() - t0, (result.get("text") or "")[:160])
        return result.get("text")

    # --- публичные методы

    def describe_event(self, frames: list[np.ndarray], event: dict, rule: dict) -> Optional[str]:
        """Описание «что видно на кадрах» в 2–3 предложениях по 1–3 кадрам (BGR) события и шаблону из правила.

        event — словарь нарушения (description, timecode_sec, type_code), rule — запись RULES[type_code]
        (name, sign). Возвращает текст или None (модель недоступна, таймаут, пустой/нерелевантный ответ —
        тогда в ответе API остаётся шаблонное описание).
        """
        if not self.available() or not self.load():
            return None
        try:
            images = _prepare_images(frames)
            if not images:
                return None
            template = str(event.get("description") or "").strip()
            tc = event.get("timecode_sec")
            when = f", момент {float(tc):.1f} с от начала видео" if isinstance(tc, (int, float)) else ""
            user_text = (
                f"{len(images)} кадр(а) с камеры в кассовом помещении банка{when}. "
                f"Правило зафиксировало событие «{rule.get('name', '')}». "
                f"Наблюдаемый признак: {rule.get('sign', '')}. "
                f"Шаблонное описание: «{template}».\n\n"
                "Перепиши это описание в 2–3 предложениях на русском языке: что именно видно на кадрах "
                "(где находится кассир, что лежит на столе, состояние сейфа, двери и экрана, есть ли другие люди). "
                "Правила: не противоречь шаблону и не добавляй того, чего на кадрах нет; не делай юридических "
                "выводов и не используй слова «похитил», «украл», «мошенничество», «виновен»; не описывай "
                "внешность, лица, пол и возраст людей — называй их только по роли (кассир, клиент, посторонний). "
                "Ответ — только текст описания, без заголовков и списков."
            )
            raw = self._generate(images, user_text, MAX_NEW_TOKENS, _env_float("VLM_TIMEOUT_SEC", 20.0))
            return sanitize_description(raw, template)
        except Exception as exc:  # noqa: BLE001
            log.exception("vlm: describe_event failed: %s", exc)
            return None

    def checklist(self, frames: list[np.ndarray], questions: list[str]) -> dict[str, Optional[bool]]:
        """Вопросы «да/нет» по кадрам — страховка от неизвестных типов нарушений (ПЛАН 5.1 п. 4б, 5.4).

        Например: ["На столе кассира есть стакан, бутылка или еда?", "В помещении больше двух человек?",
        "Дверца сейфа открыта?"]. Возвращает {вопрос: True | False | None}; None — модель недоступна,
        таймаут или ответ не распознан как «да»/«нет». По политике вердикта (ПЛАН 5.4) ответ «да» сам по себе
        НЕ создаёт событие: нужно «да» на ≥ 2 опорных кадрах подряд и подтверждение детекторами.
        Пока нигде не вызывается.
        """
        answers: dict[str, Optional[bool]] = {q: None for q in questions}
        if not questions or not self.available() or not self.load():
            return answers
        try:
            images = _prepare_images(frames)
            if not images:
                return answers
            timeout = _env_float("VLM_TIMEOUT_SEC", 20.0)
            for q in questions:
                user_text = (
                    f"{len(images)} кадр(а) с камеры в кассовом помещении банка. Вопрос: {q}\n"
                    "Ответь одним словом: «да» или «нет». Если по кадрам нельзя сказать уверенно — ответь «нет»."
                )
                answers[q] = parse_yes_no(self._generate(images, user_text, 8, timeout))
        except Exception as exc:  # noqa: BLE001
            log.exception("vlm: checklist failed: %s", exc)
        return answers


_describer: Optional[Describer] = None
_describer_lock = threading.Lock()


def get_describer() -> Describer:
    """Единственный экземпляр на процесс (воркер грузит модель один раз)."""
    global _describer
    with _describer_lock:
        if _describer is None:
            _describer = Describer()
        return _describer


def describe_violations(video_path: str, violations: list[dict], info: Any = None) -> None:
    """Подменяет `description` у нарушений текстом VLM по кадрам события. Ничего не возвращает, не бросает.

    Для каждого нарушения берёт до MAX_FRAMES сырых кадров по `evidence_frames[*].timestamp_sec`
    (без рамок и подписей) через pipeline.video.read_frame_at. Шаблон сохраняется в `description_template`.
    Бюджет: не более VLM_MAX_EVENTS событий (по умолчанию 5) и не более VLM_BUDGET_SEC секунд суммарно
    (по умолчанию 60) — чтобы не сорвать лимит 5 минут на видео. Если модель недоступна — выход сразу.
    """
    try:
        if not violations:
            return
        d = get_describer()
        if d.backend == "llamacpp" and not _env_flag("VLM_DESCRIBE", False):
            # малая модель (2B) нужна страховочной сетке для «да/нет»; свободный русский текст от неё — только по
            # явному VLM_DESCRIBE=1 (риск неточных описаний выше, чем у 7B)
            return
        if not d.available():
            log.debug("vlm: слой выключен или нет весов (%s) — описания шаблонные", d.model_path)
            return
        if not d.load():
            return
        from .rules import RULES
        from .video import read_frame_at

        max_events = max(0, _env_int("VLM_MAX_EVENTS", 5))
        budget = _env_float("VLM_BUDGET_SEC", 60.0)
        t0 = time.time()
        done = 0
        for v in violations[:max_events]:
            if time.time() - t0 > budget:
                log.warning("vlm: исчерпан бюджет %.0f с, описано %d из %d событий", budget, done, len(violations))
                break
            if v.get("source") == "safety_net":
                continue   # описание страховочной сетки — фиксированный шаблон с фактами проверки, не переписываем
            try:
                rule = RULES.get(v.get("type_code", ""), {})
                times = [f.get("timestamp_sec") for f in (v.get("evidence_frames") or [])]
                times = [float(t) for t in times if isinstance(t, (int, float))][:MAX_FRAMES]
                if not times and isinstance(v.get("timecode_sec"), (int, float)):
                    times = [float(v["timecode_sec"])]
                frames = [fr for fr in (read_frame_at(video_path, t, info) for t in times) if fr is not None]
                if not frames:
                    continue
                text = d.describe_event(frames, v, rule)
                if text:
                    v["description_template"] = v.get("description", "")
                    v["description"] = text
                    done += 1
            except Exception as exc:  # noqa: BLE001 — одно событие не должно ломать остальные
                log.exception("vlm: событие %s @ %s пропущено: %s", v.get("type_code"), v.get("timecode_sec"), exc)
        log.info("vlm: описано %d/%d событий за %.1f с", done, len(violations), time.time() - t0)
    except Exception as exc:  # noqa: BLE001 — пайплайн никогда не падает из-за VLM
        log.exception("vlm: describe_violations failed: %s", exc)


__all__ = ["Describer", "describe_violations", "get_describer", "sanitize_description", "parse_yes_no", "weights_present",
           "backend", "gguf_files", "llama_bin", "server_healthy"]
