# Полный прогон на Windows: окружение → веса → признаки по всем видео → e2e-тесты → commit + push результатов.
# Запуск: двойной клик по scripts\run_all.bat  или  powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1
# Нужны: Python 3.11–3.13 (python.org, с галочкой "Add python.exe to PATH"), git с доступом к репозиторию, датасет в ..\open\videos.
$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}
$ServiceDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ServiceDir
$TotalSw = [Diagnostics.Stopwatch]::StartNew()
$StepSw = [Diagnostics.Stopwatch]::StartNew()
$TotalSteps = 9
$WithVlm = ($env:WITH_VLM -eq "1")   # WITH_VLM=1 — дополнительно поставить зависимости VLM и скачать Qwen2.5-VL-7B (~16 ГБ)

function Step($n, $title, $desc) {
    $script:StepSw.Restart()
    Write-Host ""
    Write-Host ("=" * 78) -ForegroundColor Cyan
    Write-Host ("  ШАГ $n/$TotalSteps  $title    [" + (Get-Date -Format "HH:mm:ss") + "]") -ForegroundColor Cyan
    Write-Host ("=" * 78) -ForegroundColor Cyan
    foreach ($line in $desc) { Write-Host "  $line" -ForegroundColor Gray }
    Write-Host ""
}
function Done($msg) {
    $t = [math]::Round($script:StepSw.Elapsed.TotalSeconds)
    Write-Host ""
    Write-Host "  OK  $msg  (шаг занял $t с, всего прошло $([math]::Round($TotalSw.Elapsed.TotalMinutes, 1)) мин)" -ForegroundColor Green
}
function Fail($msg) {
    Write-Host ""
    Write-Host "  ОШИБКА: $msg" -ForegroundColor Red
    throw $msg
}

Write-Host ""
Write-Host "ПОЛНЫЙ ПРОГОН СЕРВИСА КОНТРОЛЯ КАССЫ (ForteBank, Datathon 26)" -ForegroundColor Yellow
Write-Host "Папка сервиса: $ServiceDir"
Write-Host "План: 1) Python  2) окружение и зависимости  3) веса детектора  4) проверка датасета"
Write-Host "      5) признаки по всем видео (самое долгое)  6) офлайн-оценка правил  7) e2e-тесты"
Write-Host "      8) VLM (только если WITH_VLM=1)  9) commit + push результатов"
Write-Host "Ориентир по времени: 20-40 минут (+ до часа с WITH_VLM=1). Ноутбук будет нагружен и греться - это нормально."

# ---------------------------------------------------------------- 1
Step 1 "Поиск Python" @("Ищу Python 3.11-3.13 через лаунчер 'py' или команду 'python'.")
$py = $null
foreach ($cand in @(@("py", "-3.12"), @("py", "-3.11"), @("py", "-3.13"), @("python", $null), @("python3", $null))) {
    try {
        $pyArgs = @(); if ($cand[1]) { $pyArgs += $cand[1] }; $pyArgs += "--version"
        $v = & $cand[0] @pyArgs 2>&1
        if ("$v" -match "Python 3\.1[1-3]") { $py = $cand; Write-Host "  найден: $v  (команда: $($cand -join ' '))"; break }
        else { Write-Host "  $($cand -join ' '): $v - не подходит" }
    } catch { Write-Host "  $($cand -join ' '): нет" }
}
if (-not $py) { Fail "Не найден Python 3.11-3.13. Установите с python.org (отметьте 'Add python.exe to PATH') и запустите снова." }
Done "Python найден"

# ---------------------------------------------------------------- 2
Step 2 "Виртуальное окружение и зависимости" @(
    "Создаю .venv (если нет) и ставлю пакеты из requirements.txt и requirements-dev.txt:",
    "fastapi/uvicorn (сервис), opencv (видео), numpy, onnxruntime (детектор), pillow, pytest, openpyxl (разметка).",
    "Первый раз это 2-5 минут и ~400 МБ загрузки. Ниже pip печатает каждый пакет: Collecting (ищет), Downloading",
    "(качает, с прогресс-баром), Installing collected packages (ставит). Если долго висит на Downloading - просто медленная сеть."
)
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "  создаю .venv ..."
    $pyArgs = @(); if ($py[1]) { $pyArgs += $py[1] }; $pyArgs += @("-m", "venv", ".venv")
    & $py[0] @pyArgs
    if ($LASTEXITCODE -ne 0) { Fail "не удалось создать .venv" }
} else { Write-Host "  .venv уже есть - переиспользую" }
$venvpy = Join-Path $ServiceDir ".venv\Scripts\python.exe"
Write-Host "  обновляю pip ..."
& $venvpy -m pip install --upgrade pip --disable-pip-version-check
Write-Host "  ставлю зависимости ..."
& $venvpy -m pip install --disable-pip-version-check -r requirements.txt -r requirements-dev.txt
if ($LASTEXITCODE -ne 0) { Fail "pip install не удался (см. текст выше; чаще всего - нет интернета или старый Python)" }
& $venvpy -c "import cv2, numpy, onnxruntime; print('  проверка импорта: opencv', cv2.__version__, '| numpy', numpy.__version__, '| onnxruntime', onnxruntime.__version__)"
if ($LASTEXITCODE -ne 0) { Fail "пакеты установились, но не импортируются" }
Done "зависимости готовы"

# ---------------------------------------------------------------- 3
Step 3 "Веса детектора YOLOX-m" @(
    "Скачиваю yolox_m.onnx (97 МБ) с GitHub-релиза Megvii YOLOX в service\models и проверяю sha256.",
    "Если файл уже есть и контрольная сумма сходится - пропускаю."
)
& $venvpy scripts\get_models.py
if ($LASTEXITCODE -ne 0) { Fail "не удалось скачать веса детектора (нужен интернет до github.com)" }
Done "веса на месте"

# ---------------------------------------------------------------- 4
Step 4 "Проверка датасета" @("Жду папку <репозиторий>\open\videos с 42 файлами .mov и <репозиторий>\open\open_labels.xlsx.")
if (-not (Test-Path "..\open\videos")) { Fail "Нет папки open\videos рядом с service\ - скопируйте датасет (видео + open_labels.xlsx) в <репозиторий>\open\" }
$nvid = (Get-ChildItem "..\open\videos" -Filter *.mov).Count
$sizeMb = [math]::Round(((Get-ChildItem "..\open\videos" -Filter *.mov | Measure-Object Length -Sum).Sum / 1MB))
Write-Host "  видео найдено: $nvid ($sizeMb МБ)"
if (Test-Path "..\open\open_labels.xlsx") { Write-Host "  разметка open_labels.xlsx: есть" } else { Write-Host "  разметка open_labels.xlsx: НЕТ - сводка будет без сравнения с разметкой" -ForegroundColor Yellow }
if ($nvid -eq 0) { Fail "в open\videos нет .mov-файлов" }
Done "датасет на месте"

# ---------------------------------------------------------------- 5
Step 5 "Признаки по всем видео" @(
    "Для каждого видео: декодирование -> детекция людей и объектов YOLOX на 3 к/с -> треки и признак 'внутри кассы'",
    "-> якоря сцены по медианному кадру (стол, сейф, ноутбук, дверь) -> признаки состояний на 2 к/с",
    "-> текущий пайплайн (вердикт). Результат: service\results\perception\*.json, states\*.json, verdicts.json, summary.md.",
    "Кадры и картинки НЕ сохраняются. По каждому видео ниже будут строки прогресса; оценка оставшегося времени - в них же."
)
$ErrorActionPreference = "Continue"
& $venvpy scripts\collect_features.py --videos ..\open\videos --labels ..\open\open_labels.xlsx --out results --skip-items
if ($LASTEXITCODE -ne 0) { Write-Host "  !! collect_features завершился с кодом $LASTEXITCODE - см. results\run.log; продолжаю со следующими шагами" -ForegroundColor Yellow }
Done "признаки собраны"

# ---------------------------------------------------------------- 6
Step 6 "Офлайн-оценка правил по всем видео" @(
    "scripts\eval_rules.py: прогоняет analyze_video по всем 42 видео из только что собранных признаков (без детектора),",
    "печатает таблицу: вердикт против разметки, события, ошибка таймкода, список ложных и пропусков.",
    "Ориентир на Mac (17.10): TP=11 FN=11 FP=0 TN=20. Результат сохраняется в results\eval.json и results\eval.txt."
)
& $venvpy scripts\eval_rules.py --json results\eval.json 2>&1 | Tee-Object -FilePath results\eval.txt
Done "оценка правил записана"

# ---------------------------------------------------------------- 7
Step 7 "E2E-тесты сервиса (без Docker)" @(
    "pytest в режиме -v: каждый тест печатается отдельной строкой с PASSED/FAILED/SKIPPED.",
    "Поднимается настоящий сервер на случайном порту, видео гоняются через API, пайплайн - напрямую по датасету.",
    "Ожидаемо: все passed, один xfailed (test_full_recall - полнота по всем нарушениям пока не достигнута).",
    "Лог сохраняется в results\pytest.log. Это 2-6 минут."
)
$env:E2E_SKIP_DOCKER = "1"
$env:FEATURES_CACHE = (Join-Path $ServiceDir "results")
& $venvpy -m pytest -v -ra -p no:cacheprovider 2>&1 | Tee-Object -FilePath results\pytest.log
Write-Host "  pytest завершился с кодом $LASTEXITCODE (0 = всё прошло)"
Done "тесты отработали"

# ---------------------------------------------------------------- 8
Step 8 "VLM: зависимости и веса (опционально)" @(
    "Только если переменная WITH_VLM=1: pip install -r requirements-vlm.txt и python scripts\get_vlm.py",
    "(Qwen2.5-VL-7B-Instruct, Apache-2.0, ~16 ГБ, нужна видеокарта с 8+ ГБ или 4-бит режим VLM_4BIT=1).",
    "Включается в сервисе переменной VLM_ENABLED=1 в .env. Без WITH_VLM шаг пропускается."
)
if ($WithVlm) {
    & $venvpy -m pip install --disable-pip-version-check -r requirements-vlm.txt
    if ($LASTEXITCODE -ne 0) { Write-Host "  !! зависимости VLM не установились" -ForegroundColor Yellow }
    & $venvpy scripts\get_vlm.py
    if ($LASTEXITCODE -ne 0) { Write-Host "  !! веса VLM не скачались" -ForegroundColor Yellow } else { Done "VLM готова" }
} else { Write-Host "  пропущено (WITH_VLM не задан)" }

# ---------------------------------------------------------------- 9
Step 9 "Commit и push результатов" @(
    "Добавляю в git только папку service\results, делаю commit, подтягиваю свежий main (pull --rebase) и пушу в origin main.",
    "Если push не пройдёт (нет доступа), ниже будет подсказка - тогда выполните 'git push origin main' вручную."
)
$hostName = $env:COMPUTERNAME
$stamp = Get-Date -Format "yyyy-MM-dd HH:mm"
Write-Host "  git add results"
& git add results
Write-Host "  git commit"
& git -c core.quotepath=false commit -m "results: прогон на $hostName $stamp (Windows)"
Write-Host "  git pull --rebase origin main"
& git pull --rebase origin main
Write-Host "  git push origin main"
& git push --progress origin main
if ($LASTEXITCODE -ne 0) { Write-Host "  !! push не удался: проверьте доступ к репозиторию и выполните 'git push origin main' вручную" -ForegroundColor Yellow }
else { Done "результаты в origin/main" }

Write-Host ""
Write-Host ("=" * 78) -ForegroundColor Green
Write-Host ("  ГОТОВО за $([math]::Round($TotalSw.Elapsed.TotalMinutes, 1)) мин. Результаты: service\results (summary.md - сводка, pytest.log - тесты, run.log - журнал)") -ForegroundColor Green
Write-Host ("=" * 78) -ForegroundColor Green
