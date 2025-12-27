@echo off
chcp 65001 > nul
echo ========================================
echo   Korean Calligraphy - Start All Servers
echo ========================================
echo.

cd /d "%~dp0"

echo [INFO] Starting backend server...
start "Backend Server" cmd /k "cd /d %~dp0backend && call venv\Scripts\activate.bat && python main.py"

timeout /t 2 > nul

echo [INFO] Starting frontend server...
start "Frontend Server" cmd /k "cd /d %~dp0frontend && npm run dev"

echo.
echo ========================================
echo   Servers started!
echo ========================================
echo.
echo   Backend:  http://localhost:8000
echo   Frontend: http://localhost:5173
echo.
echo   Press Ctrl+C in each window to stop.
echo ========================================

timeout /t 3
