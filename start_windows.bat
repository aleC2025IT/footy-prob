@echo off
SETLOCAL
python --version >NUL 2>&1
IF ERRORLEVEL 1 (
  echo Serve Python 3.10+ installato. Scaricalo da https://www.python.org/downloads/
  pause
  exit /b 1
)
python -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app\streamlit_app.py