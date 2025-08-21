# experiments/run_experiment.py


import sys
import os
import numpy as np
from logging import config

# Add the parent of "src" to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.meso.meso4 import Meso4Model
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
from data.data_loader import get_data_generators
from utils.dvc_utils import get_dvc_dataset_version

def main():

    # === Load config ===
    config = ConfigLoader("src/config/config.yaml")

    REAL_DIR = config.get("data.REAL_DIR")
    FAKE_DIR = config.get("data.FAKE_DIR")
    DATA_DIR = config.get("data.DATA_DIR")

    EPOCHS = config.get("train.EPOCHS")
    BATCH_SIZE = config.get("train.BATCH_SIZE")

    MODEL_MESO4_WEIGHT =  config.get("model.MODEL_MESO4_WEIGHT")
    MODEL_MESO4_FULL = config.get("model.MODEL_MESO4_FULL")


    EXPERIMENT_NAME = config.get("mlflow.EXPERIMENT_NAME_MESO")
    
    MODEL_PATH = config.get("output.MODELS_PATH")
    OUTPUT_DIR = config.get("output.OUTPUT_DIR_MESONET")
    IMAGE_SIZE = tuple(config.get("data.IMAGE_SIZE"))
    CLASSES = config.get("data.CLASSES")  # e.g., ['real', 'fake']
    RUN_NAME = config.get("mlflow.MESO4_RUN_NAME")
    MODEL_NAME = config.get("mlflow.MESO4_MODEL_NAME")
    
    # Step 1: Load Data & Data augmentation
    train_gen, val_gen = get_data_generators(data_dir=DATA_DIR,
        target_size=IMAGE_SIZE,  
        batch_size=BATCH_SIZE,  
        classes=CLASSES  # real = 0, fake = 1
    )

    # Step 2:  Model
    model = Meso4Model()

    # === Callbacks ===
    os.makedirs(os.path.dirname(MODEL_MESO4_WEIGHT), exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        filepath=MODEL_MESO4_WEIGHT,  # must end in .weights.h5 if save_weights_only=True
        save_best_only=True,
        save_weights_only=True,
        monitor="val_accuracy",
        mode="max",
        verbose=1,
    )

    early_stop_cb = EarlyStopping(
        monitor="val_accuracy", patience=3, restore_best_weights=True, verbose=1
    )

    # === Load dataset version from DVC ===
    dataset_version = get_dvc_dataset_version("dataset.dvc")
    
    # === Train ===
    history = model.train(
        train_gen= train_gen, 
        val_gen=val_gen, 
        batch_size=BATCH_SIZE, 
        epochs=EPOCHS,
        callbacks=[checkpoint_cb, early_stop_cb]
        )

    # === Evaluate on validation set (for promotion criteria) ===
    # Get predictions
    y_probs = model.predict(val_gen, steps=len(val_gen), verbose=1).ravel()
    y_pred = (y_probs > 0.5).astype("int32")  # get predictions as 0 or 1
    y_val = val_gen.classes  # get true labels from validation generator


    metrics = {
    "val_accuracy": float(accuracy_score(y_val, y_pred)),
    "val_precision": float(precision_score(y_val, y_pred, zero_division=0)),
    "val_recall": float(recall_score(y_val, y_pred)),
    "val_f1_score": float(f1_score(y_val, y_pred)),
    "val_auc_score": float(roc_auc_score(y_val, y_probs))
    }

    # Round metrics to avoid precision issues
    metrics = {k: round(v, 4) for k, v in metrics.items()}

    # === Save final weights ===
    model.save(MODEL_MESO4_WEIGHT)
    model.save(MODEL_MESO4_FULL)

    # === MLflow Run ===
    params={
        "model": str(model),
        "model_architecture": str(model),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "model_save_path": MODEL_MESO4_WEIGHT,
        "checkpoint_callback": str(checkpoint_cb),
        "early_stopping_callback": str(early_stop_cb),
        "MODEL_PATH": MODEL_PATH,
        "real_dir": REAL_DIR,
        "fake_dir": FAKE_DIR,
        "image_size": IMAGE_SIZE,  # IMAGE_SIZE
        "dataset_version": dataset_version,
        "data_dir": DATA_DIR,
        "train_generator": str(train_gen),
        "val_generator": str(val_gen),
        "train_generator_samples": train_gen.samples,
        "val_generator_samples": val_gen.samples,
        "train_generator_classes": train_gen.classes,
        "val_generator_classes": val_gen.classes,
        "train_generator_class_indices": train_gen.class_indices,
        "val_generator_class_indices": val_gen.class_indices,
        "train_generator_image_shape": train_gen.image_shape,
        "val_generator_image_shape": val_gen.image_shape,
        "train_generator_batch_size": train_gen.batch_size,
    }
    tags={"script": "run_experiment", "stage": "training", "use_case": "Deepfake Detection", "model": "Meso4", "dataset_version": dataset_version}

    # Convert all non-string parameters to string for MLflow logging
    params = {k: str(v) if not isinstance(v, (int, float, str)) else v for k, v in params.items()}


   # MLflow run
    start_mlflow_run_with_logging( 
        experiment_name=EXPERIMENT_NAME, 
        run_name= RUN_NAME, 
        model_name=MODEL_NAME,
        params=params, 
        tags=tags, 
        history=history, 
        model=model, 
        train_gen=train_gen,  # Use train_gen for X_train
        val_gen=val_gen,  # Use val_gen for y_val
        run_prefix="meso4_experiment_run",
        metrics=metrics,
        output_dir= OUTPUT_DIR  # Directory to save MLflow logs
        )
    

if __name__ == "__main__":
    main()
