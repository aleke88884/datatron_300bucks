# datatron_300bucks

Datathon 26 · кейс ForteBank — локальный сервис, который по видеозаписи кассы находит нарушения кассовой дисциплины
и выдаёт вердикт, таймкод, кадры-доказательства и ссылку на пункт нормативного акта РК.

- [`service/`](service/README.md) — сервис: FastAPI + пайплайн правил T1–T13 + веб-интерфейс, Docker, smoke-тест.
- [`ПЛАН_ForteBank.md`](ПЛАН_ForteBank.md) — разбор ТЗ, таксономия нарушений, нормативная база, архитектура, план работ.
- [`docs/INDEX.md`](docs/INDEX.md) — документы к сдаче: архитектура и модели, источники правил, типы нарушений, ограничения, питч.
- `Кейс_ForteBank.docx` — исходное задание.

Состояние (`release/final`, 18.10.2026): на 42 обучающих роликах TP 17 / FN 5 / FP 0 / TN 20 (precision 100 %, полнота 77 %,
медиана ошибки таймкода 0.6 с). Пороги подобраны на тех же роликах, отложенной выборки нет; ограничения — `docs/LIMITATIONS.md`.

Датасет (`open/`) и кадры из него в репозиторий не входят: по условиям кейса видео нельзя публиковать.
Ключ API хранится в `service/.env` (не коммитится, см. `service/.env.example`).

Запуск:

```bash
cd service
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/get_models.py   # веса YOLOX-m и RTMPose-m (один раз)
cp .env.example .env   # задать API_KEY (python3 deploy/gen_api_key.py)
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
```
