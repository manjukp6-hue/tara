@echo off
title TARA AI Core
echo ==============================================================================
echo   Starting TARA AI Core Production Engine
echo ==============================================================================
python -m pip install -r requirements.txt
start http://127.0.0.1:7860
python app.py
pause
