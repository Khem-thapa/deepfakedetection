from fastapi import FastAPI
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api.apis.predict import router as predict_router

app = FastAPI(
    title="Deepfake Detection API",
    description="API for detecting deepfake images using MesoNet-4 and EfficientNet models",
    version="1.0.0",
    docs_url="/docs"
)

app.include_router(predict_router, prefix="/predict", tags=["Prediction"])

@app.get("/")
async def root():
    return {"message": "Deepfake Detection API is running"}
