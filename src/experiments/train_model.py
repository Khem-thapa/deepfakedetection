
# experiments/run_experiment.py


import sys
import os

# Add the parent of "src" to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.dataloader import DataLoader
from models.meso4 import Meso4Model
from utils.config_loader import ConfigLoader 
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from utils.mlflow_util import start_mlflow_run_with_logging

def main():

    # === Load config ===
    config = ConfigLoader("src/config/config.yaml")

    REAL_DIR = config.get("data.REAL_DIR")
    FAKE_DIR = config.get("data.FAKE_DIR")
    EPOCHS = config.get("train.EPOCHS")
    BATCH_SIZE = config.get("train.BATCH_SIZE")
    MODEL_SAVE_PATH =  config.get("output.MODEL_SAVE_PATH")
    MODEL_PATH = config.get("output.MODELS_PATH")
    
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
    history = model.train(X_train, y_train, X_val, y_val, epochs=EPOCHS, batch_size=BATCH_SIZE)
    
   # metrics from history
    train_acc = history.history['accuracy'][-1]
    val_acc = history.history['val_accuracy'][-1]
    train_loss = history.history['loss'][-1]
    val_loss = history.history['val_loss'][-1]
    print(f"Training Accuracy: {train_acc}, Validation Accuracy: {val_acc}")
    print(f"Training Loss: {train_loss}, Validation Loss: {val_loss}")


    # validate metrics
    print("Validation Metrics:")
    print(f"train_Accuracy: {train_acc}")
    print(f"val_Accuracy: {val_acc}")
    print(f"train_Loss: {train_loss}")
    print(f"val_Loss: {val_loss}")
    
    metrics = {
    "train_acc": train_acc,
    "val_acc": val_acc,
    "train_loss": train_loss,  
    "val_loss": val_loss
    }

    # Save metrics to JSON file
    import json
    os.makedirs("results/train", exist_ok=True)
    with open("results/train/metrics.json", "w") as f:
        json.dump(metrics, f)

    # === Save final weights (optional if checkpoint saves best) ===
    model.save(MODEL_SAVE_PATH)
    

if __name__ == "__main__":
    main()
