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
from data.data_loader import get_data_generators

def main():

    # === Load config ===
    config = ConfigLoader("src/config/config.yaml")

    REAL_DIR = config.get("data.REAL_DIR")
    FAKE_DIR = config.get("data.FAKE_DIR")
    EPOCHS = config.get("train.EPOCHS")
    BATCH_SIZE = config.get("train.BATCH_SIZE")
    MODEL_WEIGHT_PATH =  config.get("output.MODEL_WEIGHT_PATH")
    MODEL_FULL_PATH = config.get("output.MODEL_FULL_PATH")
    EXPERIMENT_NAME = config.get("mlflow.EXPERIMENT_NAME")
    MODEL_PATH = config.get("output.MODELS_PATH")
    DATA_DIR = config.get("data.DATA_DIR")
    OUTPUT_DIR = config.get("output.OUTPUT_DIR_MESONET")
    # Step 1: Load Data & Data augmentation
    # Use data generators for training and validation
    train_gen, val_gen = get_data_generators(data_dir=DATA_DIR,
        target_size=(256, 256),  # IMAGE_SIZE
        batch_size=BATCH_SIZE,  # BATCH_SIZE
        classes=['real', 'fake']  # real = 0, fake = 1
    )
    # loader = DataLoader(REAL_DIR, FAKE_DIR)
    # X_train, X_val, y_train, y_val = loader.load_data()

    # Step 2:  Model
    model = Meso4Model()

    # === Callbacks ===
    os.makedirs(os.path.dirname(MODEL_WEIGHT_PATH), exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        filepath=MODEL_WEIGHT_PATH,  # must end in .weights.h5 if save_weights_only=True
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
    from utils.dvc_utils import get_dvc_dataset_version
    dataset_version = get_dvc_dataset_version("dataset.dvc")


    # === MLflow Run ===
    run_name="Meso4_Run"
    params={
        "model": str(model),
        "model_architecture": str(model),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "model_save_path": MODEL_WEIGHT_PATH,
        "checkpoint_callback": str(checkpoint_cb),
        "early_stopping_callback": str(early_stop_cb),
        "MODEL_PATH": MODEL_PATH,
        "real_dir": REAL_DIR,
        "fake_dir": FAKE_DIR,
        "image_size": (256, 256),  # IMAGE_SIZE
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

 
    # === Train ===
    history = model.train(train_gen= train_gen, val_gen=val_gen, batch_size=BATCH_SIZE, epochs=EPOCHS)
    
    # val_accuracy = model.evaluate(X_val, y_val)[1]

    # === Predict on validation set ===
    y_probs = model.predict(val_gen, verbose=1)
    y_pred = (y_probs > 0.5).astype("int32").flatten()  # get predictions as 0 or 1
    y_val = val_gen.classes  # get true labels from validation generator

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
    model.save(MODEL_WEIGHT_PATH)

    # Save the model to the specified path
    model.save(MODEL_FULL_PATH)

   # MLflow run
    start_mlflow_run_with_logging( 
        experiment_name=EXPERIMENT_NAME, 
        run_name= run_name, 
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
