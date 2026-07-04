@echo off
echo Starting PPE Backend...

start cmd /k "cd /d A:\PPE && uvicorn api:app --reload"

timeout /t 3

echo Starting React Frontend...

start cmd /k "cd /d A:\PPE\frontend && npm run dev"

timeout /t 2

start http://localhost:5173

echo PPE Sentinel Started Successfully!
pause