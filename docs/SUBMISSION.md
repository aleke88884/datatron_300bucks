# Пакет сдачи (раздел 8 ТЗ)

Что передаём жюри, как это собрать и как проверить. Поля `<...>` заполнить перед отправкой.

## 1. Что именно передаём

| Элемент | Значение | Где взять |
|---|---|---|
| Публичный URL веб-интерфейса | `https://<host>/` | `service/deploy/cloudflared.md` (туннель) или адрес GPU-сервера |
| Публичный URL API | `https://<host>/api/v1` | тот же хост; эндпоинты ниже |
| Ключ доступа (заголовок `X-API-Key`) | `<ключ>` | `python service/deploy/gen_api_key.py` → `service/.env` |
| Инструкция по запуску и проверке | этот файл + `service/README.md` | — |
| Описание архитектуры и моделей (с лицензиями) | `docs/ARCHITECTURE.md` | — |
| Описание источников и сопоставления правил | `docs/RULES_SOURCES.md` | — |
| Перечень типов нарушений | `docs/VIOLATION_TYPES.md` | — |
| Перечень ограничений и известных ошибок | `docs/LIMITATIONS.md` | — |

Текст для письма/формы жюри (заполнить):

```
Веб-интерфейс: https://<host>/
API:           https://<host>/api/v1  (POST /analyze, GET /jobs/{job_id}; заголовок X-API-Key)
Ключ:          <ключ>
Проверка:      curl -X POST https://<host>/api/v1/analyze -H "X-API-Key: <ключ>" -F "file=@video_017.mov"
               curl -H "X-API-Key: <ключ>" https://<host>/api/v1/jobs/<job_id>
Документация:  docs/ (ARCHITECTURE, RULES_SOURCES, VIOLATION_TYPES, LIMITATIONS), service/README.md
Обработка видео выполняется локально на нашем сервере; внешние AI-сервисы не используются; лица не
распознаются; загруженные видео удаляются сразу после анализа.
```

## 2. Как сгенерировать ключ

```bash
cd service
python3 deploy/gen_api_key.py            # создаст/обновит API_KEY в service/.env и напечатает ключ
python3 deploy/gen_api_key.py --show     # только показать текущий ключ из .env
```

Ключ — `secrets.token_urlsafe(32)` (43 символа URL-safe base64). После смены ключа сервис нужно перезапустить
(в Docker — `docker compose up -d`, переменные читаются из `.env`). Ключ в git не попадает (`service/.env` в
`.gitignore`).

## 3. Запуск

Вариант А — GPU-сервер (рекомендуется, не зависит от ноутбука):

```bash
cd service
python3 deploy/gen_api_key.py
python3 scripts/get_models.py                       # веса YOLOX в service/models/ (97 МБ, один раз)
docker compose -f deploy/docker-compose.gpu.yml up --build -d
curl -s http://localhost:8000/health
```

Вариант Б — своя машина (CPU) + туннель (резерв):

```bash
cd service
python3 deploy/gen_api_key.py
docker compose up --build -d                        # веса YOLOX скачиваются при сборке; или uvicorn, см. service/README.md
deploy/run_tunnel.sh 8000                           # печатает https://....trycloudflare.com
```

Подробно о туннеле и ngrok — `service/deploy/cloudflared.md`. Туннель только проксирует HTTP; видео
обрабатывается локально.

## 4. Проверка curl-ом по ТЗ

```bash
BASE=https://<host>        # или http://localhost:8000
KEY=<ключ>

# 1. здоровье (без ключа)
curl -s $BASE/health

# 2. отправка видео (дословно по ТЗ) → {"job_id": "..."} , HTTP 202, < 2 с
curl -X POST $BASE/api/v1/analyze -H "X-API-Key: $KEY" -F "file=@video_017.mov"

# 3. статус/результат: processing → done (verdict, violations[...]) или failed
curl -H "X-API-Key: $KEY" $BASE/api/v1/jobs/<job_id>

# 4. без ключа / с неверным ключом → 401
curl -s -o /dev/null -w '%{http_code}\n' -X POST $BASE/api/v1/analyze -F "file=@video_017.mov"
curl -s -o /dev/null -w '%{http_code}\n' -H "X-API-Key: wrong" $BASE/api/v1/jobs/abc
```

Всё это одним скриптом: `service/deploy/healthcheck.sh <путь к .mov> [BASE_URL]` — проверяет `/health`, 401,
отправляет видео, ждёт `done`, печатает вердикт и типы, валидирует схему.

Локальный smoke-тест на датасете: `service/scripts/smoke.sh` (порт 8010; Х-11 → `violation` T3, Х-3 → `no_violation`).
Офлайн-оценка правил по всем 42 видео (кеш признаков из `results/`): `python scripts/eval_rules.py`.

## 5. Чек-лист перед сдачей (ПЛАН §8.1)

- [ ] `curl -X POST …/api/v1/analyze -H "X-API-Key: …" -F "file=@X-22.mov"` возвращает `job_id` за < 2 с.
- [ ] Без ключа / с неверным ключом — 401.
- [ ] `GET /api/v1/jobs/{id}` → `processing` → `done` с JSON ровно по схеме ТЗ; `evidence_frames` — JPEG base64 (1–3 шт.), `timecode_sec` — число.
- [ ] Видео без нарушений → `verdict: "no_violation"`, `violations: []`.
- [ ] 142-секундное видео (Х-14) обрабатывается < 4 мин (верхняя оценка на CPU: ≈125 с; живой путь на M4 — 6 с на 17-секундное видео).
- [ ] Контейнер работает с отключённым сетевым доступом (`tests/test_docker.py`, `--network none`).
- [ ] В UI — статус, вердикт, типы, таймкоды, кадры, описание, документ + пункт + ссылка.
- [ ] В документации: архитектура и модели (с лицензиями), источники и сопоставление, перечень типов, перечень ограничений, инструкция по запуску — `docs/` + `service/README.md`.
- [ ] Лица не распознаются; тестовые видео не сохраняются дольше, чем нужно для ответа (`KEEP_UPLOADS=false`) — написано явно в `ARCHITECTURE.md` §3 и `LIMITATIONS.md` §1.
- [ ] Публичный URL открывается из другой сети (проверить с телефона через мобильный интернет).
- [ ] Машина с сервисом не уйдёт в сон (ноутбук: отключить сон при закрытой крышке / `caffeinate` на macOS); туннель запущен в отдельном окне и не закрыт.
- [ ] Ключ в письме жюри совпадает с `API_KEY` в `service/.env` на запущенном сервисе.
- [ ] Нагрузка: 5 видео подряд + 2 параллельно — все `done` в лимит (`tests/test_api_e2e.py`).
- [ ] Офлайн-оценка перед сдачей: `python scripts/eval_rules.py` — 0 ложных на чистых (на 0.3.0: 11/22 найдено, FP 0).
- [ ] Если включён VLM (`VLM_ENABLED=1`): веса в `models/vlm/`, GPU виден в контейнере, в логе `vlm: модель ... загружена`.

## 6. Что сдаём по данным

- Датасет (`open/`) и кадры из него **не публикуются** (условие кейса). В репозитории — только JSON с
  числовыми признаками (`service/results/`).
- Собственные видео с другого угла (`open/own/`, см. `OWN_VIDEOS.md`) — не в git; результат прогона — в
  `INDEX.md`/питче.
