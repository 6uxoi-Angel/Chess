@echo off
py -3.13 -c "import chess, pygame" >nul 2>&1
if errorlevel 1 (
    py -3.13 -m pip install -r "%~dp0requirements.txt"
    if errorlevel 1 (
        pause
        exit /b 1
    )
)
py -3.13 "%~dp0app.py"
