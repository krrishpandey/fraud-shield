@echo off
rem FraudShield desktop app launcher (Windows). Double-click to run.
setlocal
cd /d "%~dp0"

where uv >nul 2>nul
if errorlevel 1 (
  echo Installing uv, the Python package manager, one time only...
  powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
  set "PATH=%USERPROFILE%\.local\bin;%PATH%"
)

rem Keep the Python environment out of OneDrive and other synced folders.
if "%UV_PROJECT_ENVIRONMENT%"=="" set "UV_PROJECT_ENVIRONMENT=%USERPROFILE%\.venvs\fraudshield"

rem With an NVIDIA GPU we install the Laya model runtime; without one the app uses cached answers.
where nvidia-smi >nul 2>nul
if errorlevel 1 (
  echo No NVIDIA GPU found: running with cached model answers.
  uv sync --python 3.12 --no-dev || goto :fail
) else (
  echo NVIDIA GPU found: installing the Laya runtime. The first run downloads about 3 GB.
  uv sync --python 3.12 --no-dev --extra ml || goto :fail
)

if not exist "web\dist\index.html" (
  where npm >nul 2>nul
  if errorlevel 1 (
    echo The console UI is not built and Node.js is not installed. Ask a teammate for web\dist.
    goto :fail
  )
  pushd web
  call npm ci && call npm run build || (popd & goto :fail)
  popd
)

uv run --no-dev python -m fraudshield.desktop %*
goto :eof

:fail
echo.
echo FraudShield could not start. See the messages above.
pause
exit /b 1
