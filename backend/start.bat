@echo off
chcp 65001 > nul
echo ========================================
echo   Korean Calligraphy Backend Server
echo ========================================
echo.

cd /d "%~dp0"

if not exist "venv" (
    echo [ERROR] venv folder not found. Please install first.
    pause
    exit /b 1
)

echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

echo [INFO] Starting server... (http://localhost:8000)
echo [INFO] Press Ctrl+C to stop.
echo.

python main.py

pause
