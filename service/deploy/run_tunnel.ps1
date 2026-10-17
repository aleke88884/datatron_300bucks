# Публичный URL для локального сервиса (Windows): Cloudflare quick tunnel (по умолчанию) или ngrok.
# Использование (PowerShell, из папки service или любой):
#   .\deploy\run_tunnel.ps1                 # порт 8000, cloudflare
#   .\deploy\run_tunnel.ps1 -Port 8010      # если сервис на 8010
#   .\deploy\run_tunnel.ps1 -Mode ngrok
# Печатает публичный URL и строки для письма жюри; туннель работает, пока окно открыто (Ctrl+C — остановить).
# Обработка видео остаётся локальной: туннель только проксирует HTTP к localhost:PORT.
param(
    [int]$Port = 8000,
    [ValidateSet("cloudflare", "ngrok")][string]$Mode = "cloudflare"
)
$ErrorActionPreference = "Stop"
$ServiceDir = Split-Path -Parent $PSScriptRoot
$Key = ""
$EnvFile = Join-Path $ServiceDir ".env"
if (Test-Path $EnvFile) {
    $line = Get-Content $EnvFile | Where-Object { $_ -match '^\s*API_KEY\s*=' } | Select-Object -First 1
    if ($line) { $Key = ($line -split '=', 2)[1].Trim().Trim('"').Trim("'") }
}
if (-not $Key) { $Key = "<API_KEY из service\.env>" }

try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 5 "http://localhost:$Port/health" | Out-Null }
catch { Write-Error "сервис не отвечает на http://localhost:$Port/health — сначала запустите его (uvicorn / docker compose up -d)"; exit 1 }

function Show-Handout([string]$Url) {
    Write-Host ""
    Write-Host "================ публичный адрес ================"
    Write-Host "UI:   $Url/"
    Write-Host "API:  $Url/api/v1"
    Write-Host "Ключ: $Key   (заголовок X-API-Key)"
    Write-Host "curl -X POST $Url/api/v1/analyze -H `"X-API-Key: $Key`" -F `"file=@video_017.mov`""
    Write-Host "curl -H `"X-API-Key: $Key`" $Url/api/v1/jobs/<job_id>"
    Write-Host "================================================="
    Write-Host "(адрес действует, пока работает этот процесс; Ctrl+C — остановить)"
    Write-Host ""
}

if ($Mode -eq "cloudflare") {
    if (-not (Get-Command cloudflared -ErrorAction SilentlyContinue)) {
        Write-Error "cloudflared не установлен: winget install --id Cloudflare.cloudflared (см. deploy\cloudflared.md)"; exit 1
    }
    $log = Join-Path $env:TEMP ("cloudflared-" + [guid]::NewGuid().ToString("N") + ".log")
    $proc = Start-Process -FilePath "cloudflared" -ArgumentList @("tunnel", "--url", "http://localhost:$Port", "--no-autoupdate") `
        -RedirectStandardError $log -RedirectStandardOutput ($log + ".out") -NoNewWindow -PassThru
    $url = $null
    for ($i = 0; $i -lt 60 -and -not $url; $i++) {
        Start-Sleep -Seconds 1
        foreach ($f in @($log, ($log + ".out"))) {
            if (Test-Path $f) {
                $m = Select-String -Path $f -Pattern 'https://[a-z0-9-]+\.trycloudflare\.com' | Select-Object -First 1
                if ($m) { $url = $m.Matches[0].Value; break }
            }
        }
        if ($proc.HasExited) { Write-Error "cloudflared завершился; лог: $log"; Get-Content $log; exit 1 }
    }
    if (-not $url) { Write-Error "не дождались адреса trycloudflare.com; лог: $log"; Get-Content $log; exit 1 }
    Show-Handout $url
    try { Get-Content -Path $log -Wait } finally { if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force } }
}
else {
    if (-not (Get-Command ngrok -ErrorAction SilentlyContinue)) {
        Write-Error "ngrok не установлен: winget install ngrok.ngrok, затем ngrok config add-authtoken <токен> (см. deploy\cloudflared.md)"; exit 1
    }
    $proc = Start-Process -FilePath "ngrok" -ArgumentList @("http", "$Port", "--log=stdout") -NoNewWindow -PassThru -RedirectStandardOutput (Join-Path $env:TEMP "ngrok.log")
    $url = $null
    for ($i = 0; $i -lt 30 -and -not $url; $i++) {
        Start-Sleep -Seconds 1
        try {
            $t = (Invoke-RestMethod -UseBasicParsing "http://127.0.0.1:4040/api/tunnels").tunnels | Where-Object { $_.public_url -like "https*" } | Select-Object -First 1
            if ($t) { $url = $t.public_url }
        } catch {}
        if ($proc.HasExited) { Write-Error "ngrok завершился (authtoken задан?)"; exit 1 }
    }
    if (-not $url) { Write-Error "не дождались адреса ngrok (панель http://127.0.0.1:4040)"; exit 1 }
    Show-Handout $url
    try { Wait-Process -Id $proc.Id } finally { if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force } }
}
