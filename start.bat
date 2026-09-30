@echo off
setlocal EnableExtensions
rem ============================================================
rem LottoMax AI - local one-click launcher (Windows double-click)
rem Backend 127.0.0.1:%LOTTOMAX_PORT% (default 8000) + frontend localhost:5173.
rem Closing this window or pressing a key stops both servers.
rem ============================================================
cd /d "%~dp0"
title LottoMax AI
if "%LOTTOMAX_PORT%"=="" set "LOTTOMAX_PORT=8000"
set "BE_PORT=%LOTTOMAX_PORT%"
set "FE_PORT=5173"
set "LOGDIR=%~dp0.launcher-logs"
if not exist "%LOGDIR%" mkdir "%LOGDIR%"

where python >nul 2>nul || (echo [ERROR] Python not found. Install Python 3.10-3.12 first. & goto :fail)
where node   >nul 2>nul || (echo [ERROR] Node.js not found. Install Node.js 18+ first. & goto :fail)

rem ---- ports must be free ----
call :port_busy %BE_PORT% && (echo [ERROR] Port %BE_PORT% is already in use. See README "Running locally". & goto :fail)
call :port_busy %FE_PORT% && (echo [ERROR] Port %FE_PORT% is already in use. See README "Running locally". & goto :fail)

rem ---- first-time setup ----
if not exist "be\venv\Scripts\python.exe" (
  echo [setup] Creating be\venv and installing requirements ^(first run only; TensorFlow is large^)...
  python -m venv be\venv || (echo [ERROR] Could not create the virtual environment. & goto :fail)
  "be\venv\Scripts\python.exe" -m pip install -r be\requirements.txt || (echo [ERROR] pip install failed ^(TensorFlow needs Python 3.10-3.12^). & goto :fail)
)
if not exist "fe\node_modules" (
  echo [setup] Running npm install in fe\ ^(first run only^)...
  pushd fe
  call npm install || (popd & echo [ERROR] npm install failed. & goto :fail)
  popd
)

rem ---- backend ----
echo [1/2] Starting backend on http://127.0.0.1:%BE_PORT% ...
pushd be
start "" /b cmd /c ""venv\Scripts\python.exe" app.py > "%LOGDIR%\backend.log" 2>&1"
popd

set /a N=0
:wait_be
"be\venv\Scripts\python.exe" -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:%BE_PORT%/', timeout=2)" >nul 2>nul && goto :be_ok
set /a N+=1
if %N% GEQ 60 (
  echo [ERROR] Backend did not respond on http://127.0.0.1:%BE_PORT%/ within 60 seconds.
  echo ---- backend log ----
  type "%LOGDIR%\backend.log"
  goto :fail
)
timeout /t 1 /nobreak >nul
goto :wait_be
:be_ok
echo       backend is up.

rem ---- frontend ----
echo [2/2] Starting frontend on http://localhost:%FE_PORT% ...
set "VITE_API_URL=http://localhost:%BE_PORT%"
pushd fe
start "" /b cmd /c "npm run dev > "%LOGDIR%\frontend.log" 2>&1"
popd

set /a N=0
:wait_fe
"be\venv\Scripts\python.exe" -c "import urllib.request,sys; urllib.request.urlopen('http://localhost:%FE_PORT%/', timeout=2)" >nul 2>nul && goto :fe_ok
set /a N+=1
if %N% GEQ 60 (
  echo [ERROR] Frontend did not respond on http://localhost:%FE_PORT% within 60 seconds.
  echo ---- frontend log ----
  type "%LOGDIR%\frontend.log"
  goto :fail
)
timeout /t 1 /nobreak >nul
goto :wait_fe
:fe_ok
echo       frontend is up.

start "" "http://localhost:%FE_PORT%"
echo.
echo LottoMax AI is running: http://localhost:%FE_PORT%
echo Press any key here (or close this window) to stop both servers.
pause >nul
call :cleanup
exit /b 0

:fail
call :cleanup
echo.
pause
exit /b 1

rem ---- helpers ----
:port_busy
netstat -ano | findstr /R /C:":%1 .*LISTENING" >nul
exit /b %errorlevel%

:cleanup
echo Stopping LottoMax AI...
for %%P in (%FE_PORT% %BE_PORT%) do (
  for /f "tokens=5" %%I in ('netstat -ano ^| findstr /R /C:":%%P .*LISTENING"') do taskkill /F /T /PID %%I >nul 2>nul
)
exit /b 0
