@echo off
title SAFESIGHT Launcher

echo ==========================================
echo        SAFESIGHT AI Platform
echo ==========================================
echo.

:: ------------------------------------------
:: Change to Project Directory
:: ------------------------------------------
cd /d A:\PPE

:: ------------------------------------------
:: Check Backend
:: ------------------------------------------
if not exist api.py (
    echo ERROR: api.py not found!
    pause
    exit
)

:: ------------------------------------------
:: Check Frontend
:: ------------------------------------------
if not exist frontend (
    echo ERROR: frontend folder not found!
    pause
    exit
)

echo Starting FastAPI Backend...
start "SAFESIGHT Backend" cmd /k "cd /d A:\PPE && uvicorn api:app --reload"

timeout /t 5 >nul

echo Starting React Frontend...
start "SAFESIGHT Frontend" cmd /k "cd /d A:\PPE\frontend && npm run dev"

timeout /t 5 >nul

echo Opening Dashboard...
start http://localhost:5173/

timeout /t 2 >nul

echo Opening AI Assistant...
start http://localhost:5173/assistant

echo.
echo ==========================================
echo SAFESIGHT Started Successfully!
echo ==========================================
echo.
echo Dashboard   : http://localhost:5173/
echo AI Assistant: http://localhost:5173/assistant
echo Backend API : http://127.0.0.1:8000/docs
echo.

pause