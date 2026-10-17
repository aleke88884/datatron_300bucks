# Сервис контроля кассовой дисциплины по видео (Datathon 26, кейс ForteBank)

Локальный веб-сервис + API: принимает `.mov`-запись кассы, выдаёт вердикт `violation` / `no_violation`,
список событий с таймкодом, кадрами-доказательствами (JPEG base64) и ссылкой на пункт нормативного акта РК.
Всё выполняется локально: без облачных AI-сервисов, без распознавания лиц, видео удаляется после анализа.

Структура:

```
service/
├── app/            FastAPI: эндпоинты, X-API-Key, очередь (один воркер-процесс, таймаут 240 с)
├── pipeline/       анализ видео: video.py (OpenCV/FFmpeg), detectors.py (правила T1–T13), rules.py (нормативка)
├── static/         фронтенд — одна страница, без сборки
├── scripts/smoke.sh
├── Dockerfile, docker-compose.yml, requirements.txt, .env.example
```

## Запуск локально

Нужен Python 3.12. Системный ffmpeg не требуется (используется ffmpeg из `imageio-ffmpeg`/OpenCV).

```bash
cd service
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env            # и задайте API_KEY (python3 -c 'import secrets;print(secrets.token_hex(16))')
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Интерфейс: http://localhost:8000 · Swagger: http://localhost:8000/docs · здоровье: `GET /health`.

## Запуск в Docker (CPU)

```bash
cd service
cp .env.example .env            # задайте API_KEY
docker compose up --build -d
curl -s http://localhost:8000/health
```

После сборки образа интернет не нужен (веса моделей добавляются в образ на этапе сборки).

## API (раздел 10 ТЗ)

Все запросы к `/api/v1/*` требуют заголовок `X-API-Key`; без ключа или с неверным — `401`.

| Метод | Путь | Назначение |
|---|---|---|
| POST | `/api/v1/analyze` | multipart-поле `file` (.mov, до 300 МБ) → `{"job_id": "..."}` (HTTP 202) |
| GET | `/api/v1/jobs/{job_id}` | статус `processing` / `done` / `failed` и результат |
| GET | `/api/v1/rules` | справочник типов нарушений и нормативных пунктов |
| GET | `/health` | без ключа; состояние воркера и очереди |

Пример (из ТЗ):

```bash
curl -X POST http://localhost:8000/api/v1/analyze \
     -H "X-API-Key: <ключ>" -F "file=@video_017.mov"
```

```bash
curl -H "X-API-Key: <ключ>" http://localhost:8000/api/v1/jobs/3f2a9c1e
```

Ответ при завершении (поля ТЗ + расширения `type_code`, `confidence`, `rule`, `end_sec`, `meta`):

```json
{
  "job_id": "3f2a9c1e",
  "status": "done",
  "verdict": "violation",
  "violations": [
    {
      "type": "Потеря видеоконтроля (освещение выключено)",
      "timecode_sec": 16.06,
      "description": "В 16.1 с средняя яркость кадра упала с 139 до 19 (на 86 %) ...",
      "evidence_frames": [
        {"timestamp_sec": 15.06, "image_base64": "/9j/4AAQ..."},
        {"timestamp_sec": 16.06, "image_base64": "/9j/4AAQ..."},
        {"timestamp_sec": 18.06, "image_base64": "/9j/4AAQ..."}
      ],
      "type_code": "T9",
      "confidence": 0.99,
      "end_sec": 25.6,
      "rule": {
        "document": "Правила организации охраны и устройства помещений банков ... (постановление Правления НБРК от 24.02.2020 № 14)",
        "clause": "п. 16 пп. 6)",
        "url": "https://adilet.zan.kz/rus/docs/V2000020075"
      }
    }
  ],
  "meta": {"duration_sec": 25.6, "fps": 29.883, "processing_time_sec": 4.1, "pipeline_version": "0.1.0-skeleton"}
}
```

Статусы: `processing` (с полем `progress: {stage, percent}`), `done`, `failed` (с полем `error`).
Если обработка не укладывается в `JOB_TIMEOUT_SEC` (240 с), воркер-процесс убивается, задание получает `failed`.

## Smoke-тест

```bash
cd service && scripts/smoke.sh
```

Поднимает сервер на порту 8010, отправляет `open/videos/Х-11.mov` (ожидается `violation`, T9) и `open/videos/Х-3.mov`
(ожидается `no_violation`), печатает JSON, проверяет `401` без ключа и соответствие схеме ТЗ.

## Пайплайн и как добавлять правила

`pipeline/analyze_video(path, progress) -> {"verdict", "violations", "meta"}` — единственная точка входа для воркера.

1. `pipeline/video.py` — декодирование (OpenCV, бэкенд FFmpeg), ряд яркости, выборка кадра по времени, JPEG/base64.
2. `pipeline/context.py` — `VideoContext` (общие данные видео: яркость; слоты `detections`, `tracks`, `states`, `zones`
   для слоя восприятия) и `Event` (что возвращает правило).
3. `pipeline/detectors.py` — правила `detect_tN(ctx) -> list[Event]` по таксономии T1–T13. Реализован **T9**
   (освещение выключено: падение средней яркости кадра > 40 % относительно предыдущих 3 с, удержание > 1 с).
   Остальные — заглушки с докстрингами: признак, ожидаемые входы, примеры видео. Список `DETECTORS` задаёт, что включено.
4. `pipeline/rules.py` — справочник `type_code → название, признак, документ, пункт, url` строго по разделу 4
   ПЛАН_ForteBank.md (Правила № 231 и № 14 НБРК). Выдуманных пунктов нет; перед сдачей сверить формулировки на adilet.
5. `pipeline/draw.py` — баннер-подпись на кадре (кириллица через Pillow + системный TTF; в Docker — DejaVu) и рамки.

Чтобы добавить правило: заполнить тело `detect_tN`, вернуть `Event(type_code, timecode_sec, description,
confidence, evidence_times=[...], evidence_captions=[...])`. Таймкод = момент перехода состояния (ПЛАН 5.4), описание —
только наблюдаемое (ПЛАН 4.4). Событие попадает в ответ при `confidence >= MIN_CONFIDENCE` (0.5).
Тяжёлые модели загружать в `pipeline.warmup()` — он вызывается один раз в воркер-процессе.

## Переменные окружения (`.env`)

| Переменная | По умолчанию | Назначение |
|---|---|---|
| `API_KEY` | — (обязательна) | ключ для `X-API-Key` |
| `MAX_UPLOAD_MB` | 300 | лимит размера файла |
| `JOB_TIMEOUT_SEC` | 240 | таймаут обработки одного видео |
| `KEEP_UPLOADS` | false | хранить ли видео после анализа (по ТЗ — нет) |
| `ALLOWED_EXTENSIONS` | `.mov,.mp4` | допустимые расширения |
| `DATA_DIR` | `service/data` | временные файлы загрузок |

## Ограничения каркаса и известные ошибки

- Реализовано только правило T9 по яркости кадра. На 42 обучающих видео оно срабатывает на Х-11 (нарушение, таймкод 16.1 с
  при разметке 18 с) **и на Х-12 (размечено как «без нарушения»)**: там кассир при пустом столе сама выключает свет
  у выключателя на 32.4 с — штатный уход. По одной яркости эти случаи неразличимы, поэтому T9 нужно комбинировать
  с состояниями сцены (экран ноутбука активен / ценности на столе / кассир уходит в темноте), когда появится слой детекций.
- Остальные правила T1–T8, T10–T13 — заглушки, возвращают пустой список.
- Описания шаблонные; VLM-слой для описаний не подключён.
- Результаты заданий хранятся в памяти процесса API до перезапуска; история проверок по ТЗ не требуется.
- Один воркер: видео обрабатываются по очереди (это сознательно: предсказуемое время и память).
