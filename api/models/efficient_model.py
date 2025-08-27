import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow.keras.models import load_model as tf_load_model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def load_model():
    """
    Load the EfficientNet model for deepfake detection
    """
    model_path = "models/efficientnet/efficient_model.h5"    
    # model = tf_load_model(model_path)
    model = tf.keras.models.load_model(model_path, compile=False)
    return model

def predict_image(model, preprocessed_image):
    """
    Make prediction using the EfficientNet model
    """
    try:
        prediction = model.predict(preprocessed_image)
        return prediction[0][0]  # Return the probability
    except Exception as e:
        raise Exception(f"Error predicting with EfficientNet: {str(e)}")
