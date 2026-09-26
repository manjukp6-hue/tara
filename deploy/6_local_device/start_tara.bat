@echo off
title TARA AI Core - Local Device Engine
echo ==============================================================================
echo   TARA AI Core - Starting Local Production Engine
echo ==============================================================================
cd /d "%~dp0\..\.."
python -m pip install -r requirements.txt
echo Starting TARA Web Server on http://127.0.0.1:7860 ...
start http://127.0.0.1:7860
python app.py
pause
