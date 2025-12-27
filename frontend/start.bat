@echo off
chcp 65001 > nul
echo ========================================
echo   Korean Calligraphy Frontend Server
echo ========================================
echo.

cd /d "%~dp0"

if not exist "node_modules" (
    echo [INFO] node_modules not found. Installing packages...
    npm install
)

echo [INFO] Starting dev server... (http://localhost:5173)
echo [INFO] Press Ctrl+C to stop.
echo.

npm run dev

pause
