# datatron_300bucks

Datathon 26 · кейс ForteBank — локальный сервис, который по видеозаписи кассы находит нарушения кассовой дисциплины
и выдаёт вердикт, таймкод, кадры-доказательства и ссылку на пункт нормативного акта РК.

- [`service/`](service/README.md) — сервис: FastAPI + пайплайн правил T1–T13 + веб-интерфейс, Docker, smoke-тест.
- [`ПЛАН_ForteBank.md`](ПЛАН_ForteBank.md) — разбор ТЗ, таксономия нарушений, нормативная база, архитектура, план работ.
- `Кейс_ForteBank.docx` — исходное задание.

Датасет (`open/`) и кадры из него в репозиторий не входят: по условиям кейса видео нельзя публиковать.
Ключ API хранится в `service/.env` (не коммитится, см. `service/.env.example`).

Запуск:

```bash
cd service
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env   # задать API_KEY
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
```
