@echo off
setlocal
cd /d "%~dp0"
set "PATHSNAP_DATA=%CD%\data"
set "OMP_NUM_THREADS=1"
set "OPENBLAS_NUM_THREADS=1"
if not exist ".venv\Scripts\python.exe" (
  py -3.12 -m venv .venv || goto failed
)
".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto failed
echo Preparing GPS data...
".venv\Scripts\python.exe" -m member1_data.prepare || goto failed
echo Training five models...
".venv\Scripts\python.exe" -m member2_models.train || goto failed
".venv\Scripts\python.exe" show_results.py || goto failed
start "" "OPEN_CHECKPOINT.html"
pause
exit /b 0
:failed
echo Setup or training stopped. Check the error above. This launcher requires Python 3.12.
pause
exit /b 1
