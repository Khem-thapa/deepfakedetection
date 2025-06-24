import sys
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import argparse

def predict_image(args):
    """Predicts whether an image is real or fake using a pre-trained EfficientNet model.
    Args:
        img_path (str): Path to the image file.
    """
    if args.image_path:
        model = load_model('models/efficientnet_model.h5') 
        img = image.load_img(args.image_path, target_size=(256,256))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = model.predict(img_array)
        
        label = "Fake" if prediction[0][0] > 0.5 else "Real"
        print(f"Prediction: {label} ({prediction[0][0]:.4f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Custom EfficientNet Deepfake Detection Prediction")
    parser.add_argument("--image-path", type=str, help="Path to an image for single prediction")
    args = parser.parse_args()

    predict_image(args)
   
