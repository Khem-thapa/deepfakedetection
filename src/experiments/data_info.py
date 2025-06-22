
# experiments/run_experiment.py


import sys
import os

# Add the parent of "src" to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.dataloader import DataLoader
# from models.meso4 import Meso4Model
from models.meso4_optimized import Meso4_Opt_Model
from utils.config_loader import ConfigLoader 
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


def main():

    # === Load config ===
    config = ConfigLoader("src/config/config.yaml")

    REAL_DIR = config.get("data.REAL_DIR")
    FAKE_DIR = config.get("data.FAKE_DIR")
    EPOCHS = config.get("train.EPOCHS")
    BATCH_SIZE = config.get("train.BATCH_SIZE")
    # MODEL_WEIGHT_PATH =  config.get("output.MODEL_WEIGHT_PATH")
    MODEL_OPT_FULL_PATH = config.get("output.MODEL_OPT_FULL_PATH")

    
    # Step 1: Load Data
    loader = DataLoader(REAL_DIR, FAKE_DIR)
    X_train, X_val, y_train, y_val = loader.load_data()

    import numpy as np
    print(f"[INFO] Training set size: {len(X_train)}, Validation set size: {len(X_val)}")
    # Check if the data is loaded correctly
    print(f"[DEBUG] X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"[DEBUG] X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")    
    # Check if the data is balanced
    print(f"[DEBUG] Training set label distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"[DEBUG] Validation set label distribution: {dict(zip(*np.unique(y_val, return_counts=True)))}")
    # Check if the data is normalized
    print(f"[DEBUG] X_train min: {X_train.min()}, max: {X_train.max()}")    
    print(f"[DEBUG] X_val min: {X_val.min()}, max: {X_val.max()}")

    # print top 5 images from training set
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.imshow(X_train[i])
        plt.title(f"Label: {y_train[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()
 
    
if __name__ == "__main__":
    main()
