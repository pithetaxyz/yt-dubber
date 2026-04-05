@echo off
if "%~1"=="" (
    echo Usage: run.bat <youtube_url> [--upload]
    echo.
    echo Examples:
    echo   run.bat https://youtu.be/xxxxx
    echo   run.bat https://youtu.be/xxxxx --upload
    echo   run.bat https://youtu.be/xxxxx --whisper medium --voice en-US-JennyNeural
    exit /b 1
)
py -3.12 "%~dp0dubber.py" %*
pause
