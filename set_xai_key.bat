@echo off
echo ========================================
echo SET XAI API KEY
echo ========================================
echo.

if "%1"=="" (
    echo Usage: set_xai_key.bat "your_api_key_here"
    echo.
    echo Or run interactively:
    python set_xai_key.py
    pause
    exit /b 1
)

python set_xai_key.py "%1"

echo.
echo ========================================
echo To make it permanent in PowerShell:
echo $env:XAI_API_KEY="your_key_here"
echo ========================================
echo.
pause

