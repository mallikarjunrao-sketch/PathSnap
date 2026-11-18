@echo off
setlocal
cd /d "%~dp0"
if not defined PATHSNAP_DATA set "PATHSNAP_DATA=%CD%\data"
if not defined PATHSNAP_SPLIT set "PATHSNAP_SPLIT=student_test"
set "OMP_NUM_THREADS=1"
set "OPENBLAS_NUM_THREADS=1"
if not exist ".venv\Scripts\python.exe" (
  py -3.12 -m venv .venv || goto failed
)
".venv\Scripts\python.exe" -c "import fastapi, uvicorn, sklearn, numpy, scipy, pandas, joblib; assert sklearn.__version__ == '1.8.0'" >nul 2>&1
if errorlevel 1 (
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto failed
)
echo On Laptop 2 enter the address for this laptop's Wi-Fi or hotspot adapter.
".venv\Scripts\python.exe" -c "import socket; print('\n'.join('  http://' + ip + ':8000' for ip in sorted(set(socket.gethostbyname_ex(socket.gethostname())[2])) if not ip.startswith('127.')))"
echo If multiple addresses appear, use ipconfig to identify the active adapter.
echo Keep this window open. Current trip folder: %PATHSNAP_DATA%\%PATHSNAP_SPLIT%
".venv\Scripts\python.exe" -m uvicorn member4_demo.api:app --host 0.0.0.0 --port 8000
if errorlevel 1 goto failed
exit /b 0
:failed
echo Server setup or launch failed. Check the error above. Use Python 3.12 on Laptop 1.
pause
exit /b 1
