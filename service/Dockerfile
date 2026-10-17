# CPU-вариант. Модели/веса добавлять в образ на этапе сборки, чтобы контейнер работал без интернета.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 fonts-dejavu-core curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY pipeline ./pipeline
COPY static ./static
COPY scripts/get_models.py ./scripts/get_models.py

# Веса детектора YOLOX-m и модели позы RTMPose-m (Apache-2.0) скачиваются на этапе сборки: контейнер работает без интернета.
# Если веса уже лежат в service/models/, они копируются и повторно не скачиваются (get_models проверяет sha256).
# get_models.py в списке источников — «якорь»: маска models* всегда что-то находит, и сборка не падает на свежем
# клоне, где папки models/ ещё нет (*.onnx в репозитории не хранятся).
COPY scripts/get_models.py models* ./models/
RUN rm -f ./models/get_models.py && python scripts/get_models.py

# Проверка на этапе сборки: веса загружаются, детектор отрабатывает на пустом кадре, декодер видео доступен.
# Если что-то из этого сломано, образ не соберётся — а не упадёт у жюри на первом видео.
RUN python -c "import numpy as np, cv2, imageio_ffmpeg; \
from pipeline.detect import get_detector; \
d = get_detector(); d(np.zeros((848, 464, 3), np.uint8)); \
print('build check ok: detector + opencv', cv2.__version__)"

RUN useradd -m -u 10001 svc && mkdir -p /app/data && chown -R svc:svc /app
USER svc

ENV DATA_DIR=/app/data
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD curl -fsS http://127.0.0.1:8000/health || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
