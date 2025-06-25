
# experiments/run_experiment.py


import sys
import os

# Add the parent of "src" to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from data.dataloader import DataLoader
from data.data_loader import get_data_generators
# from models.meso4 import Meso4Model
from models.meso4_optimized import Meso4_Opt_Model
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
    DATA_DIR = config.get("data.DATA_DIR")
    EPOCHS = config.get("train.EPOCHS")
    BATCH_SIZE = config.get("train.BATCH_SIZE")
    # MODEL_WEIGHT_PATH =  config.get("output.MODEL_WEIGHT_PATH")
    MODEL_OPT_FULL_PATH = config.get("output.MODEL_OPT_FULL_PATH")


    # Step 1: Load Data
    # loader = DataLoader(REAL_DIR, FAKE_DIR)
    # X_train, X_val, y_train, y_val = loader.load_data()

    # Use data generators for training and validation
    train_gen, val_gen = get_data_generators(data_dir=DATA_DIR,
        target_size=(256, 256),  # IMAGE_SIZE
        batch_size=BATCH_SIZE,  # BATCH_SIZE
        classes=['real', 'fake']  # real = 0, fake = 1
    )

    # Check if the data is loaded correctly
    print(f"[DEBUG] Training set: {train_gen.class_indices}, Validation set: {val_gen.class_indices}")
    print(f"[DEBUG] Training set batch size: {train_gen.batch_size}, Validation set batch size: {val_gen.batch_size}")
    print(f"[DEBUG] Training set image size: {train_gen.image_shape}, Validation set image size: {val_gen.image_shape}")
    print(f"[DEBUG] Training set classes: {train_gen.classes}, Validation set classes: {val_gen.classes}")
    print(f"[INFO] Training set size: {train_gen.samples}, Validation set size: {val_gen.samples}") 

    # import numpy as np
    # print(f"[INFO] Training set size: {len(X_train)}, Validation set size: {len(X_val)}")
    # # Check if the data is loaded correctly
    # print(f"[DEBUG] X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    # print(f"[DEBUG] X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
    # # Check if the data is balanced
    # print(f"[DEBUG] Training set label distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    # print(f"[DEBUG] Validation set label distribution: {dict(zip(*np.unique(y_val, return_counts=True)))}")
    # # Check if the data is normalized
    # print(f"[DEBUG] X_train min: {X_train.min()}, max: {X_train.max()}")
    # print(f"[DEBUG] X_val min: {X_val.min()}, max: {X_val.max()}")



    # Step 2:  Model
    model = Meso4_Opt_Model()

    # === Callbacks ===
    # os.makedirs(os.path.dirname(MODEL_WEIGHT_PATH), exist_ok=True)

    # checkpoint_cb = ModelCheckpoint(
    #     filepath=MODEL_WEIGHT_PATH,  # must end in .weights.h5 if save_weights_only=True
    #     save_best_only=True,
    #     save_weights_only=True,
    #     monitor="val_accuracy",
    #     mode="max",
    #     verbose=1,
    # )

    early_stop_cb = EarlyStopping(
        monitor="val_accuracy", patience=3, restore_best_weights=True, verbose=1
    )

    # === Train ===
    # history = model.train(X_train, y_train, X_val, y_val, epochs=EPOCHS, batch_size=BATCH_SIZE)

    history = model.train(train_gen= train_gen, val_gen=val_gen, batch_size=BATCH_SIZE, epochs=EPOCHS)
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

    from utils.metrics_logger import log_metrics_to_json
    log_metrics_to_json(metrics, output_dir="results/train", filename="metrics.json")

    # === Save final weights (optional if checkpoint saves best) ===
    # model.save(MODEL_WEIGHT_PATH)

    model.save(MODEL_OPT_FULL_PATH)  # Save the full model

if __name__ == "__main__":
    main()
