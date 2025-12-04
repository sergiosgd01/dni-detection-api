@echo off
cd /d "%~dp0"
call venv\Scripts\activate
start "" python main.py
timeout /t 5
start "" ngrok http 8000