# Полный прогон на Windows: окружение → веса → признаки по всем видео → e2e-тесты → commit + push результатов.
# Запуск: двойной клик по scripts\run_all.bat  или  powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1
# Нужны: Python 3.11–3.13 (python.org, с галочкой "Add python.exe to PATH"), git с доступом к репозиторию, датасет в ..\open\videos.
$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}
$ServiceDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ServiceDir
Write-Host "== папка сервиса: $ServiceDir"

# --- 1. Python
$py = $null
foreach ($cand in @(@("py", "-3.12"), @("py", "-3.11"), @("py", "-3.13"), @("python", $null), @("python3", $null))) {
    try {
        $pyArgs = @(); if ($cand[1]) { $pyArgs += $cand[1] }; $pyArgs += "--version"
        $v = & $cand[0] @pyArgs 2>&1
        if ("$v" -match "Python 3\.1[1-3]") { $py = $cand; Write-Host "== $v ($($cand -join ' '))"; break }
    } catch {}
}
if (-not $py) { throw "Не найден Python 3.11-3.13. Установите с python.org (отметьте 'Add python.exe to PATH') и запустите снова." }

# --- 2. venv + зависимости
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    $pyArgs = @(); if ($py[1]) { $pyArgs += $py[1] }; $pyArgs += @("-m", "venv", ".venv")
    & $py[0] @pyArgs
    if ($LASTEXITCODE -ne 0) { throw "не удалось создать .venv" }
}
$venvpy = Join-Path $ServiceDir ".venv\Scripts\python.exe"
& $venvpy -m pip install --upgrade pip -q
& $venvpy -m pip install -q -r requirements.txt -r requirements-dev.txt
if ($LASTEXITCODE -ne 0) { throw "pip install не удался" }

# --- 3. веса детектора
& $venvpy scripts\get_models.py
if ($LASTEXITCODE -ne 0) { throw "не удалось скачать веса детектора" }

# --- 4. датасет
if (-not (Test-Path "..\open\videos")) { throw "Нет папки open\videos рядом с service\ - скопируйте датасет (видео + open_labels.xlsx) в <репозиторий>\open\" }

# --- 5. признаки по всем видео (самое долгое: 15-40 мин)
$ErrorActionPreference = "Continue"
& $venvpy scripts\collect_features.py --videos ..\open\videos --labels ..\open\open_labels.xlsx --out results
if ($LASTEXITCODE -ne 0) { Write-Host "!! collect_features завершился с ошибкой, см. results\run.log" }

# --- 6. e2e-тесты без Docker (лог в results\pytest.log); падение тестов не прерывает скрипт
$env:E2E_SKIP_DOCKER = "1"
& $venvpy -m pytest -q -p no:cacheprovider 2>&1 | Tee-Object -FilePath results\pytest.log
Write-Host "== pytest завершился с кодом $LASTEXITCODE (0 = всё прошло)"

# --- 7. commit + push результатов (только папка results)
$hostName = $env:COMPUTERNAME
$stamp = Get-Date -Format "yyyy-MM-dd HH:mm"
& git add results
& git -c core.quotepath=false commit -m "results: прогон на $hostName $stamp (Windows)"
& git pull --rebase origin main
& git push origin main
if ($LASTEXITCODE -ne 0) { Write-Host "!! push не удался: проверьте доступ к репозиторию и выполните 'git push origin main' вручную" }
Write-Host ""
Write-Host "== ГОТОВО. Результаты в service\results. Сводка: results\summary.md, тесты: results\pytest.log"
