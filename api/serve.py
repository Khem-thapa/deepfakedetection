import os
import importlib.util
import sys

def check_uvicorn_installed():
    return importlib.util.find_spec("uvicorn") is not None

def run_server():
    if not check_uvicorn_installed():
        print("❌ Uvicorn is not installed. Run: pip install uvicorn")
        sys.exit(1)

    print("🚀 Starting FastAPI server on http://0.0.0.0:8000 ...")
    os.system("uvicorn api.main:app --reload --host 0.0.0.0 --port 8000")

if __name__ == "__main__":
    run_server()