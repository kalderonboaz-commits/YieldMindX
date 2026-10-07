@echo off
REM One-click launcher: starts the local checkpoint-dashboard backend (using
REM the project's own .venv) and opens the dashboard in your default browser.
REM Double-click this file to run it.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0launch_dashboard.ps1"
if errorlevel 1 (
    echo.
    echo Launcher reported an error -- see messages above.
    pause
)
