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
    EXPERIMENT_NAME = config.get("mlflow.EXPERIMENT_NAME")
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

    # === MLflow Run ===
    run_name="Meso4_Run"
    params={
        "model": str(model),
        "model_architecture": str(model),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "model_save_path": MODEL_SAVE_PATH,
        "checkpoint_callback": str(checkpoint_cb),
        "early_stopping_callback": str(early_stop_cb),
        "data_loader": str(loader), 
        "X_train_shape": str(X_train.shape),
        "X_val_shape": str(X_val.shape), 
        "y_train_shape": str(y_train.shape),
        "y_val_shape": str(y_val.shape),
        "MODEL_PATH": MODEL_PATH,
        "real_dir": REAL_DIR,
        "fake_dir": FAKE_DIR
    }
    tags={"script": "run_experiment", "stage": "training"}

    # Convert all non-string parameters to string for MLflow logging
    params = {k: str(v) if not isinstance(v, (int, float, str)) else v for k, v in params.items()}

 
    # === Train ===
    history = model.train(X_train, y_train, X_val, y_val, epochs=EPOCHS, batch_size=BATCH_SIZE)
    
    # val_accuracy = model.evaluate(X_val, y_val)[1]

    # === Predict on validation set ===
    y_probs = model.predict(X_val)
    y_pred = (y_probs > 0.5).astype("int32").flatten()
    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred, zero_division=0)
    rec = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_probs)

    metrics = {
    "accuracy": acc,
    "precision": prec,
    "recall": rec,
    "f1_score": f1,
    "auc_score": auc
}

    # === Save final weights (optional if checkpoint saves best) ===
    model.save(MODEL_SAVE_PATH)

   # MLflow run
    start_mlflow_run_with_logging( 
        experiment_name=EXPERIMENT_NAME, 
        run_name= run_name, 
        params=params, 
        tags=tags, 
        history=history, 
        model=model, 
        X_train=X_train, 
        y_train=y_train,
        run_prefix="meso4_experiment_run",
        metrics=metrics
        )
    

if __name__ == "__main__":
    main()
