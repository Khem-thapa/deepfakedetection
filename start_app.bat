@echo off
start cmd /k "uvicorn api.main:app --host 0.0.0.0 --port 8000"
start cmd /k "streamlit run frontend/streamlit_app.py"
