# Пакет сдачи (раздел 8 ТЗ)

Что передаём жюри, как это собрать и как проверить. Поля `<...>` заполнить перед отправкой.
Состояние на 18.10.2026: TP 17 / FN 5 / FP 0 / TN 20 на 42 обучающих роликах (`eval_rules.py` перепроверен), 321 passed, 4 skipped (3 — нет Docker-демона, 1 — нет весов RTMPose в среде проверки), 1 xfailed (`test_full_recall`).

## 0. Чек-лист поставки по разделу 8 ТЗ

| # | Что требует ТЗ | Статус | Где |
|---|---|---|---|
| 1 | Публичный URL веб-интерфейса | [x] **заполнить**: `<https://dig-preference-lobby-display.trycloudflare.com>` (туннель `cloudflared`, `service/deploy/cloudflared.md`) | §1 |
| 2 | Публичный URL API | [x] **заполнить**: `<https://dig-preference-lobby-display.trycloudflare.com/api/v1>` | §1 |
| 3 | Ключ доступа `X-API-Key` | [x] сгенерирован в `service/.env` (`python3 deploy/gen_api_key.py --show` — показать; на другой машине сгенерировать там же); в письмо жюри — да, в git и документы — нет | §2 |
| 4 | Инструкция по запуску и проверке | [x] этот файл §3–4, [service/README.md](../service/README.md), `service/deploy/healthcheck.sh` | — |
| 5 | Описание архитектуры и используемых моделей (с лицензиями) | [x] [ARCHITECTURE.md](ARCHITECTURE.md) (§2 — таблица моделей и лицензий) | — |
| 6 | Описание источников и сопоставления правил | [x] [RULES_SOURCES.md](RULES_SOURCES.md) (источник → пункт → тип → признак → видео, дословные цитаты) | — |
| 7 | Перечень типов нарушений | [x] [VIOLATION_TYPES.md](VIOLATION_TYPES.md) (T1–T13, признак, примеры, статус) | — |
| 8 | Перечень ограничений и известных ошибок | [x] [LIMITATIONS.md](LIMITATIONS.md) | — |
| 9 | Питч 5 минут + 2 минуты вопросов | [x] [PITCH.md](PITCH.md); [ ] прогнать вслух с таймером, записать резервное видео демо | — |
| 10 | Устойчивость к другому ракурсу | [x] синтетика: [service/results/robustness.md](../service/results/robustness.md); [ ] собственные видео с другого угла ([OWN_VIDEOS.md](OWN_VIDEOS.md)) | — |
| 11 | История «мимикрии» купюр (для питча) | [x] [MIMICRY.md](MIMICRY.md) | — |

Метрики для письма и питча: `service/results/eval.txt` (офлайн-оценка), `service/results/live_check.json` (живой прогон Х-14 —
89 с в производственных настройках, 143.6 с на всех ядрах), `service/results/robustness.md`.
Оговорка, которую нужно произнести честно: пороги подбирались на тех же 42 роликах, отложенной выборки нет.

## 1. Что именно передаём

| Элемент | Значение | Где взять |
|---|---|---|
| Публичный URL веб-интерфейса | `https://dig-preference-lobby-display.trycloudflare.com` | `service/deploy/cloudflared.md` (туннель) или адрес GPU-сервера |
| Публичный URL API | `https://dig-preference-lobby-display.trycloudflare.com/api/v1` | тот же хост; эндпоинты ниже |
| Ключ доступа (заголовок `X-API-Key`) | `<ключ — вписать после gen_api_key.py>` | `python service/deploy/gen_api_key.py` → `service/.env` |
| Инструкция по запуску и проверке | этот файл + `service/README.md` | — |
| Описание архитектуры и моделей (с лицензиями) | [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) | — |
| Описание источников и сопоставления правил | `docs/RULES_SOURCES.md` | — |
| Перечень типов нарушений | `docs/VIOLATION_TYPES.md` | — |
| Перечень ограничений и известных ошибок | `docs/LIMITATIONS.md` | — |

Текст для письма/формы жюри (заполнить):

```
Веб-интерфейс: https://dig-preference-lobby-display.trycloudflare.com
API:           https://dig-preference-lobby-display.trycloudflare.com/api/v1  (POST /analyze, GET /jobs/{job_id}; заголовок X-API-Key)
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

Вариант Б — своя машина (CPU) + туннель (резерв). **Одной командой** (venv, зависимости, веса, ключ, сервер, туннель, адрес для письма):

```bash
bash service/deploy/start_public.sh        # Ctrl+C — остановить; повторный запуск без pip: SKIP_INSTALL=1
```

Или по шагам:

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
BASE=https://dig-preference-lobby-display.trycloudflare.com        # или http://localhost:8000
KEY=<ключ>

# 1. здоровье (без ключа)
curl -s $BASE/health

# 2. отправка видео (дословно по ТЗ) → {"job_id": "..."} , HTTP 200, < 2 с
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

- [x] `curl -X POST …/api/v1/analyze -H "X-API-Key: …" -F "file=@X-22.mov"` возвращает `job_id` за < 2 с (`test_analyze_returns_job_id_quickly`).
- [x] Без ключа / с неверным ключом — 401 (e2e-тесты).
- [x] `GET /api/v1/jobs/{id}` → `processing` → `done` с JSON ровно по схеме ТЗ; `evidence_frames` — JPEG base64 (1–3 шт.), `timecode_sec` — число.
- [x] Видео без нарушений → `verdict: "no_violation"`, `violations: []`.
- [ ] 142-секундное видео (Х-14) обрабатывается < 4 мин: на M3 8 ГБ — 143.6 с на всех ядрах, 89 с в производственных настройках (деградированный режим, вердикт верный).
- [ ] Контейнер работает с отключённым сетевым доступом (`tests/test_docker.py`, `--network none`).
- [x] В UI — статус, вердикт, типы, таймкоды, кадры, описание, документ + пункт + ссылка; прогресс по этапам cash/door/pose проверен в Chromium 18.10.
- [ ] В документации: архитектура и модели (с лицензиями), источники и сопоставление, перечень типов, перечень ограничений, инструкция по запуску — `docs/` + `service/README.md`.
- [ ] Лица не распознаются; тестовые видео не сохраняются дольше, чем нужно для ответа (`KEEP_UPLOADS=false`) — написано явно в `ARCHITECTURE.md` §3 и `LIMITATIONS.md` §1.
- [ ] Публичный URL открывается из другой сети (проверить с телефона через мобильный интернет).
- [ ] Машина с сервисом не уйдёт в сон (ноутбук: отключить сон при закрытой крышке / `caffeinate` на macOS); туннель запущен в отдельном окне и не закрыт.
- [ ] Ключ в письме жюри совпадает с `API_KEY` в `service/.env` на запущенном сервисе.
- [x] Нагрузка: 5 видео одновременно — все `done` (`test_concurrent_uploads_all_complete`); живая пачка из 3 видео — `results/batch_check.md`.
- [ ] Пакетная проверка на сервере жюри: `python scripts/batch_check.py --base http://127.0.0.1:8000 Х-47.mov Х-13.mov Х-14.mov` — провалов 0.
- [x] Офлайн-оценка перед сдачей: `python scripts/eval_rules.py` — TP 17 / FN 5 / FP 0 / TN 20 (18.10, после всех правок).
- [x] Полный `pytest`: 321 passed, 4 skipped (3 — нет Docker-демона, 1 — нет весов RTMPose в среде проверки), 1 xfailed (`test_full_recall`).
- [ ] Docker-образ собран и запущен с `--network none`: на машине с Docker Desktop `cd service && .venv/bin/python -m pytest tests/test_docker.py -v` (до сих пор образ не собирался).
- [x] Перечитать п. 71 пп. 9) № 53, п. 12 и п. 20 абз. 2 № 231 на adilet.zan.kz в браузере — 18.10, совпадают дословно.
- [ ] Повторный прогон `scripts\run_all.bat` на Windows; результат — в `results/`.
- [ ] Если включён VLM (`VLM_ENABLED=1`): веса в `models/vlm/`, GPU виден в контейнере, в логе `vlm: модель ... загружена`.

## 6. Что сдаём по данным

- Датасет (`open/`) и кадры из него **не публикуются** (условие кейса). В репозитории — только JSON с
  числовыми признаками (`service/results/`).
- Собственные видео с другого угла (`open/own/`, см. `OWN_VIDEOS.md`) — не в git; результат прогона — в
  `INDEX.md`/питче. Сейчас устойчивость подтверждена только синтетическими преобразованиями (`results/robustness.md`).
