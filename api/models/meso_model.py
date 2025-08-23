import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.meso.meso4 import Meso4Model 

def load_model():
    model = Meso4Model()
    model.load('models/meso4/meso4.weights.h5')
    print("[INFO] Model Loaded!")
    return model

def predict_image(model, image):
    y_pred = model.predict(image)
    return y_pred[0][0]
