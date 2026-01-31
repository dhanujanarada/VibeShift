@echo off
REM Setup script for VibeShift on Windows

echo ==========================================
echo 🎵 VibeShift - Automated Setup
echo ==========================================
echo.

REM Check if uv is installed
where uv >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo 📦 Installing uv package manager...
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    if %ERRORLEVEL% NEQ 0 (
        echo ❌ Failed to install uv. Please install manually.
        exit /b 1
    )
) else (
    echo ✅ uv is already installed
)

REM Install Python 3.10
echo 🐍 Ensuring Python 3.10 is available...
uv python install 3.10
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Failed to install Python 3.10
    exit /b 1
)

REM Create virtual environment and install dependencies
echo 📚 Installing dependencies...
uv sync
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Failed to install dependencies
    exit /b 1
)

echo.
echo ==========================================
echo 🎉 Setup completed successfully!
echo ==========================================
echo.
echo Next steps:
echo 1. Activate the virtual environment:
echo    .venv\Scripts\activate
echo.
echo 2. Start the application:
echo    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
echo.
pause
