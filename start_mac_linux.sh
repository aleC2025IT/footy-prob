#!/usr/bin/env bash
set -e
if ! command -v python3 &> /dev/null; then
  echo "Serve Python 3.10+ (installalo da python.org)"; exit 1
fi
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app/streamlit_app.py