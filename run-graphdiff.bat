@echo off
setlocal EnableExtensions
rem ---------------------------------------------------------------------------
rem  graphdiff launcher for Windows. Double-click this file.
rem
rem  First run: finds Python 3.11+ (the py launcher, python on PATH, or an
rem  Anaconda/Miniconda install), creates a private environment in .venv next
rem  to this file, installs graphdiff into it. Every run: starts the web app on
rem  the demo datasets and opens your browser. Close this window to stop it.
rem
rem  To use your own graphs:  run-graphdiff.bat C:\path\to\folder\of\graphs
rem ---------------------------------------------------------------------------
cd /d "%~dp0"
title graphdiff
echo.
echo  graphdiff launcher   %~dp0
echo.

set "PY="
set "CONDA="
if exist "%~dp0.venv\Scripts\python.exe" set "PY=%~dp0.venv\Scripts\python.exe"
if exist "%~dp0.venv\python.exe"         set "PY=%~dp0.venv\python.exe"
if defined PY goto :have_env

rem --- 1. a regular Python 3.11+ on this machine? -----------------------------
for %%C in ("py -3.13" "py -3.12" "py -3.11" "python" "python3") do (
    if not defined PY (
        %%~C -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1
        if not errorlevel 1 set "PY=%%~C"
    )
)
if defined PY (
    echo  using %PY%
    echo  creating a private environment in .venv  (one-time, about a minute)...
    %PY% -m venv "%~dp0.venv" || goto :fail
    set "PY=%~dp0.venv\Scripts\python.exe"
    goto :install
)

rem --- 2. Anaconda / Miniconda? use conda to make the environment -------------
for %%D in ("%USERPROFILE%\anaconda3" "%USERPROFILE%\miniconda3" "%LOCALAPPDATA%\anaconda3" "%LOCALAPPDATA%\miniconda3" "C:\ProgramData\anaconda3" "C:\ProgramData\miniconda3") do (
    if not defined CONDA if exist "%%~D\Scripts\conda.exe" set "CONDA=%%~D\Scripts\conda.exe"
)
if defined CONDA (
    echo  using Anaconda at %CONDA%
    echo  creating a private environment in .venv  (one-time, a few minutes)...
    "%CONDA%" create -y -q -p "%~dp0.venv" python=3.11 pip >nul || goto :fail
    set "PY=%~dp0.venv\python.exe"
    goto :install
)

echo  [!] No Python 3.11 or newer was found.
echo      Install it from https://www.python.org/downloads/  (tick "Add python.exe to PATH"),
echo      then double-click this file again.
echo.
pause
exit /b 1

:install
echo  installing graphdiff and its dependencies  (one-time, a few minutes)...
"%PY%" -m pip install --upgrade pip --quiet --disable-pip-version-check
"%PY%" -m pip install -e ".[viewer,plot]" --quiet --disable-pip-version-check || goto :fail
echo  done.
echo.

:have_env
"%PY%" -c "import graphdiff, fastapi, uvicorn" >nul 2>&1 || (
    echo  environment looks incomplete; reinstalling graphdiff...
    "%PY%" -m pip install -e ".[viewer,plot]" --quiet --disable-pip-version-check || goto :fail
)

if "%~1"=="" (
    echo  starting the app on the demo datasets.  Close this window to stop.
    echo.
    "%PY%" -m graphdiff.cli app --demo
) else (
    echo  starting the app on  %~1  .  Close this window to stop.
    echo.
    "%PY%" -m graphdiff.cli app --workspace "%~1"
)
echo.
pause
exit /b 0

:fail
echo.
echo  [!] Something failed above. Take a screenshot of this window, or copy its text, and send it to me.
echo.
pause
exit /b 1
