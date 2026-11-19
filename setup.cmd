@echo off
setlocal
cd /d "%~dp0"
py -3.14 -V >nul 2>&1
if errorlevel 1 (
  py install 3.14
  if errorlevel 1 goto failed
)
py -3.14 -m venv .venv
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
echo Project environment ready. PyTorch can be installed once the GPU is identified.
where nvidia-smi >nul 2>&1
if not errorlevel 1 nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
pause
exit /b 0
:failed
echo Setup stopped. Copy the error above so it can be diagnosed.
pause
exit /b 1
