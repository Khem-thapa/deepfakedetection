
import subprocess
import threading
import time
import sys

def start_api():
    """Start FastAPI server"""
    subprocess.run([
        sys.executable, "-m", "uvicorn", 
        "api.main:app", 
        "--host", "0.0.0.0", 
        "--port", "8000"
    ])

def start_frontend():
    """Start Streamlit frontend"""
    # Wait a moment for API to start
    time.sleep(3)
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", 
        "frontend/streamlit_app.py",
        "--server.port", "8501",
        "--server.address", "0.0.0.0"
    ])

if __name__ == "__main__":
    # Start API in a separate thread
    api_thread = threading.Thread(target=start_api, daemon=True)
    api_thread.start()
    
    # Start frontend in main thread
    start_frontend()
 

 