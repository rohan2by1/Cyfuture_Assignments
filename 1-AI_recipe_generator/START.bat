@echo off
REM Create a virtual environment named 'venv'
python -m venv venv

REM Activate the virtual environment
call .\venv\Scripts\activate.bat

REM Install required packages
pip install -r requirements.txt
python -m spacy download en_core_web_sm

REM Run the Python script
streamlit run main.py

REM Pause to keep the window open after execution
pause
