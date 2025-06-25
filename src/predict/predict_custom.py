import os
import sys
import argparse
import numpy as np
import pandas as pd
from PIL import Image

from sklearn.metrics import classification_report

# Add parent of src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.meso4_optimized import Meso4_Opt_Model
from models.meso4 import Meso4Model
from data.dataloader import DataLoader
from utils.config_loader import ConfigLoader

# === Image Preprocessing ===
def preprocess_image(image_path, image_size):
    try:
        print(f"[DEBUG] Image path: {image_path}")
        print(f"[DEBUG] Resize to: {image_size}")
        img = Image.open(image_path).convert("RGB")
        img = img.resize(image_size)
        img_array = np.array(img) / 255.0
        return np.expand_dims(img_array, axis=0)  # Add batch dim
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

# === Main Function ===
def main(args):
    config = ConfigLoader("src/config/config.yaml")

    IMAGE_SIZE = tuple(config.get("data.IMAGE_SIZE"))
    # CUSTOM_WEIGHTS = config.get("output.MODEL_WEIGHT_PATH")
    MODEL_FULL_PATH = config.get("output.MODEL_OPT_FULL_PATH")
    RESULTS_DIR = config.get("output.RESULT_DIR")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load custom model
    # model = Meso4_Opt_Model()
    # model.load(MODEL_FULL_PATH)
    model = Meso4Model()
    model.load(MODEL_FULL_PATH)

    if args.image_path:
        # === Predict from a single image ===
        input_image = preprocess_image(args.image_path, IMAGE_SIZE)
        if input_image is None:
            print("Failed to process image.")
            return

        prediction = model.predict(input_image)[0][0]
        predicted_label = int(prediction > 0.5)
        label_str = "Fake" if predicted_label == 1 else "Real"
        print(f"\nPrediction: {label_str} (Confidence: {prediction:.4f})")


# === CLI Arguments ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Custom Meso4 Deepfake Detection Prediction")
    parser.add_argument("--image-path", type=str, help="Path to an image for single prediction")
    args = parser.parse_args()

    main(args)
