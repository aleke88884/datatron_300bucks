# Публичный URL для жюри: Cloudflare Tunnel (основной) и ngrok (резерв)

Жюри нужны публичные URL веб-интерфейса и API. Если сервис крутится на нашей машине (ноутбук/ПК с GPU), публичный
адрес даёт туннель. **Туннель только проксирует HTTP-запросы к `localhost`; обработка видео остаётся локальной,
на нашей машине** — это не нарушает требование локальности (ПЛАН §5.5: модели выполняются внутри нашего сервиса,
внешних AI-сервисов нет). Видео через туннель проходит к нам и после анализа удаляется.

Порт: сервис по умолчанию слушает **8000** (uvicorn и `docker compose`); `scripts/smoke.sh` поднимает свой
экземпляр на 8010. Ниже везде `PORT` — тот, на котором реально запущен сервис.

## 1. Cloudflare Tunnel (quick tunnel, без аккаунта)

Установка `cloudflared`:

- macOS: `brew install cloudflared`
- Windows: `winget install --id Cloudflare.cloudflared` (или скачать `cloudflared-windows-amd64.exe` со
  страницы релизов https://github.com/cloudflare/cloudflared/releases и положить в PATH)
- Linux (deb): `curl -L -o cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb && sudo dpkg -i cloudflared.deb`

Запуск (сервис уже слушает `localhost:PORT`):

```bash
cloudflared tunnel --url http://localhost:8000
```

Через несколько секунд в логе появится строка вида

```
https://random-words-1234.trycloudflare.com
```

Это и есть публичный URL: UI — `https://.../`, API — `https://.../api/v1/analyze`, `https://.../api/v1/jobs/{id}`.
Скрипты `run_tunnel.sh` / `run_tunnel.ps1` запускают туннель и печатают URL отдельно.

Особенности quick tunnel:
- адрес случайный и **меняется при каждом перезапуске** `cloudflared` — отправлять жюри только после того, как
  туннель поднят окончательно; не перезапускать до конца проверки;
- аккаунт и домен не нужны; лимит размера тела запроса у Cloudflare — 100 МБ для бесплатного плана
  (видео датасета — единицы–десятки МБ, достаточно; если жюри пришлёт файл > 100 МБ, нужен ngrok или
  именованный туннель с планом выше);
- таймаут ответа — 100 с на запрос; наш API отвечает мгновенно (`job_id`) и опрашивается по `GET`, поэтому
  долгая обработка не упирается в таймаут;
- если нужен **постоянный адрес**: именованный туннель на своём домене через `cloudflared tunnel login` →
  `cloudflared tunnel create cashdesk` → `cloudflared tunnel route dns cashdesk cashdesk.<домен>` →
  `cloudflared tunnel run --url http://localhost:8000 cashdesk` (нужен аккаунт Cloudflare и домен на нём).

## 2. ngrok (запасной вариант)

Нужен аккаунт (бесплатный) и authtoken: https://dashboard.ngrok.com/get-started/your-authtoken

```bash
# macOS: brew install ngrok ; Windows: winget install ngrok.ngrok ; Linux: см. https://ngrok.com/download
ngrok config add-authtoken <токен>
ngrok http 8000
```

В консоли ngrok — строка `Forwarding https://xxxx.ngrok-free.app -> http://localhost:8000`. Это публичный URL.
Нюансы бесплатного плана: случайный адрес (постоянный — на платном или через `--url` с зарезервированным
доменом), межстраничное предупреждение в браузере при первом открытии UI (API-запросам с заголовком
`ngrok-skip-browser-warning: 1` оно не мешает; `curl` по ТЗ работает без него, но на всякий случай в
инструкции жюри это можно указать). Лимит тела запроса у ngrok выше, чем 100 МБ.

## 3. Что передать жюри

```
UI:   https://<адрес туннеля>/
API:  https://<адрес туннеля>/api/v1
Ключ: <API_KEY из service/.env>   (заголовок X-API-Key)
curl -X POST https://<адрес>/api/v1/analyze -H "X-API-Key: <ключ>" -F "file=@video_017.mov"
curl -H "X-API-Key: <ключ>" https://<адрес>/api/v1/jobs/<job_id>
```

Проверить снаружи перед отправкой: `deploy/healthcheck.sh <видео.mov> https://<адрес>` с другой сети
(мобильный интернет на телефоне → открыть `https://<адрес>/health`).

## 4. Пока жюри тестирует

- Машина не должна уснуть: macOS — `caffeinate -dims` в отдельном терминале; Windows — «Электропитание →
  никогда не переводить в спящий режим», ноутбук с закрытой крышкой — «при закрытии крышки: действие не требуется».
- Туннель и сервис — в отдельных окнах/сессиях (`tmux`/`screen` на Linux), не закрывать.
- Следить за `GET /health` (там `worker_alive`, размер очереди и счётчики заданий) и логом сервиса.
- Если сервис или туннель упал: перезапустить сервис (`docker compose up -d`), перезапустить туннель, **новый
  адрес отправить жюри** — поэтому лучше именованный туннель/домен или арендованный сервер (ПЛАН §5.5, вариант А).

## 5. Безопасность

- Без ключа API отвечает 401; ключ — 43 символа (`secrets.token_urlsafe(32)`), в git не хранится.
- Туннель открывает только порт сервиса; других сервисов на этом порту быть не должно.
- После завершения проверки жюри туннель остановить (Ctrl+C), ключ можно сменить (`gen_api_key.py`).
