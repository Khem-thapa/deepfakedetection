from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image
import io

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.meso_model import load_model, predict_image
from utils.preprocess import preprocess_image

router = APIRouter()

model = load_model()

@router.post("/")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        preprocessed = preprocess_image(image)
        prediction = predict_image(model, preprocessed)

        label = "Fake" if prediction > 0.5 else "Real"

        return {
            "filename": file.filename,
            "label": label,
            "confidence": float(prediction)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
