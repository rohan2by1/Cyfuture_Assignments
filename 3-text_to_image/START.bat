@echo off
REM Create a virtual environment named 'venv'
python -m venv venv

REM Activate the virtual environment
call .\venv\Scripts\activate.bat

REM Install required packages
pip install -r requirements.txt

REM Run the Python script
python main.py

REM Pause to keep the window open after execution
pause
