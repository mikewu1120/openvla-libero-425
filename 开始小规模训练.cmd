@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" -u train_qlora_4080.py
pause
