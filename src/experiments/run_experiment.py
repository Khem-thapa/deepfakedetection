# experiments/run_experiment.py


import sys
import os

# Add the parent of "src" to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.dataloader import DataLoader
from models.meso4 import Meso4Model
from utils.config_loader import ConfigLoader 
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def main():

    # === Load config ===
    config = ConfigLoader("src/config/config.yaml")

    REAL_DIR = config.get("data.REAL_DIR")
    FAKE_DIR = config.get("data.FAKE_DIR")
    EPOCHS = config.get("train.EPOCHS")
    BATCH_SIZE = config.get("train.BATCH_SIZE")
    MODEL_SAVE_PATH =  config.get("output.MODEL_SAVE_PATH")
    
    # Step 1: Load Data
    loader = DataLoader(REAL_DIR, FAKE_DIR)
    X_train, X_val, y_train, y_val = loader.load_data()

    # Step 2:  Model
    model = Meso4Model()

    # === Callbacks ===
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        filepath=MODEL_SAVE_PATH,  # must end in .weights.h5 if save_weights_only=True
        save_best_only=True,
        save_weights_only=True,
        monitor="val_accuracy",
        mode="max",
        verbose=1,
    )

    early_stop_cb = EarlyStopping(
        monitor="val_accuracy", patience=3, restore_best_weights=True, verbose=1
    )

    # === Train ===
    model.train(X_train, y_train, X_val, y_val, epochs=EPOCHS, batch_size=BATCH_SIZE)

    # === Save final weights (optional if checkpoint saves best) ===
    model.save(MODEL_SAVE_PATH)

if __name__ == "__main__":
    main()
