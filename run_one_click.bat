@echo off
echo ========================================
echo ONE-CLICK IMAGE COLORIZER
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.7 or higher
    pause
    exit /b 1
)

REM Install packages if needed
echo Installing required packages...
python install_all_packages.py

echo.
echo ========================================
echo Starting Colorization...
echo ========================================
echo.

REM Run the one-click colorizer
if "%1"=="" (
    python one_click_colorizer.py
) else (
    python one_click_colorizer.py "%1"
)

echo.
pause

