@echo off
chcp 65001 >nul
echo == Full run: venv, model weights, features for all videos, rules eval, tests, push of results. Takes 20-40 minutes.
echo == To also install the VLM (Qwen2.5-VL-7B, ~16 GB): run  set WITH_VLM=1  before this script.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_all.ps1"
echo.
echo == Exit code: %ERRORLEVEL%
pause
