@echo off
title PostureGuard Installer
cd /d "%~dp0"

echo =======================================================
echo          PostureGuard Desktop Setup & Installer
echo =======================================================
echo.

REM 1. Check or activate Python virtual environment
if not exist "venv\Scripts\python.exe" (
    echo [*] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Python not found in PATH. Please install Python 3.10+ and re-run.
        pause
        exit /b 1
    )
)

echo [*] Installing dependencies...
venv\Scripts\pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo [ERROR] Failed to install requirements.
    pause
    exit /b 1
)

REM 2. Register Startup in Windows Registry
echo [*] Enabling Windows Startup (Auto-start on boot)...
venv\Scripts\python -c "from src.desktop_app import StartupManager; StartupManager.set_enabled(True); print('Startup configured successfully!')"

echo.
echo =======================================================
echo   Installation Complete!
echo.
echo   - PostureGuard will now launch automatically on Windows boot.
echo   - It runs silently in the background with NO camera on screen.
echo   - Slouching alert triggers if slouching persists for >10s.
echo   - Pop-up dismisses automatically after 5s upright sitting.
echo =======================================================
echo.

set /p START_NOW="Do you want to start PostureGuard now? (Y/N): "
if /i "%START_NOW%"=="Y" (
    echo [*] Starting PostureGuard silently in the background...
    wscript.exe "Start_PostureGuard.vbs"
)

echo Done! You can close this window.
pause
