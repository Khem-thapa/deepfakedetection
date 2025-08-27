from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image
import io

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.meso_model import load_model as load_meso_model
from models.meso_model import predict_image as predict_meso
from models.efficient_model import load_model as load_efficient_model
from models.efficient_model import predict_image as predict_efficient
from utils.preprocess import preprocess_image, preprocess_image_ef

router = APIRouter()

# Load both models at startup
meso_model = load_meso_model()
efficient_model = load_efficient_model()

@router.post("/meso")
async def predict_meso_net(file: UploadFile = File(...)):
    """
    Endpoint for MesoNet-4 model predictions
    """
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        preprocessed = preprocess_image(image)
        prediction = predict_meso(meso_model, preprocessed)
        
        label = "Fake" if prediction > 0.5 else "Real"
        
        return {
            "filename": file.filename,
            "label": label,
            "confidence": float(prediction) * 100  # Convert to percentage
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/efficient")
async def predict_efficient_net(file: UploadFile = File(...)):
    """
    Endpoint for EfficientNet model predictions
    """
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        preprocessed = preprocess_image_ef(image)
        prediction = predict_efficient(efficient_model, preprocessed)
        
        label = "Fake" if prediction > 0.5 else "Real"
        
        return {
            "filename": file.filename,
            "label": label,
            "confidence": float(prediction) * 100  # Convert to percentage
        }
    except Exception as e:
        # raise HTTPException(status_code=500, detail=str(e))
        """
        Legacy endpoint - redirects to MesoNet endpoint
        """
    return await predict_meso_net(file)
