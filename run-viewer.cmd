@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Run setup.cmd once to create the project environment.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m streamlit run src/viewer_app.py --server.address 127.0.0.1 --server.port 8503
if errorlevel 1 pause
