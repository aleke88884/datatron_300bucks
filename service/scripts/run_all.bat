@echo off
chcp 65001 >nul
echo == Full run: venv, model weights, features for all videos, tests, push of results. Takes 20-40 minutes.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_all.ps1"
echo.
echo == Exit code: %ERRORLEVEL%
pause
