@echo off
chcp 65001 > nul
echo ========================================
echo   Korean Calligraphy - Stop All Servers
echo ========================================
echo.

echo [INFO] Stopping port 8000 (backend)...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000 ^| findstr LISTENING') do (
    taskkill /PID %%a /F 2>nul
)

echo [INFO] Stopping port 5173 (frontend)...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :5173 ^| findstr LISTENING') do (
    taskkill /PID %%a /F 2>nul
)

echo.
echo [Done] All servers stopped.
timeout /t 2
