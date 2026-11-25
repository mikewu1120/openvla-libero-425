@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" -u run_libero_4080.py --adapter training_runs/qlora_task0/adapter --output results/qlora_task0
pause
