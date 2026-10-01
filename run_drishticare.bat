@echo off
cd /d "%~dp0"
title Drishticare Fundus Quality Web App
where py >nul 2>nul
if %errorlevel%==0 (set PY=py) else (set PY=python)
if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  %PY% -m venv .venv
  if errorlevel 1 goto fail
)
echo Installing required packages...
".venv\Scripts\python.exe" -m pip install flask numpy opencv-python
if errorlevel 1 goto fail
echo Starting Drishticare...
start "" http://127.0.0.1:5000
".venv\Scripts\python.exe" app.py
goto :eof
:fail
echo Setup failed. Make sure Python 3.10+ is installed.
pause
