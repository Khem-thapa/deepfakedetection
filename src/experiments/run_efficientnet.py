# Add the parent of "src" to Python path
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.data_loader import get_data_generators
from models.efficientnet.model_builder import build_efficientnet_model

from utils.mlflow_util import start_mlflow_run_with_logging
from data.data_loader import get_data_generators


# === Load config ===

from utils.config_loader import ConfigLoader
config = ConfigLoader("src/config/config.yaml")

OUTPUT_DIR = config.get("output.OUTPUT_DIR_EFFICIENT")

EPOCHS = config.get("train.EPOCHS")
BATCH_SIZE = config.get("train.BATCH_SIZE")

INPUT_SHAPE = tuple(config.get("data.INPUT_SHAPE"))  # e.g., (256, 256, 3)
DATA_DIR = config.get("data.DATA_DIR")

OPTIMIZER = config.get("parameters.OPTIMIZER")  # e.g., 'adam'
LOSS_FUNC = config.get("parameters.LOSS_FUNC")  # e.g., 'binary_crossentropy'
IMAGE_SIZE = tuple(config.get("data.IMAGE_SIZE"))
CLASSES = config.get("data.CLASSES")  # e.g., ['real', 'fake']

# Define experiment config
EXPERIMENT_NAME = config.get("mlflow.EXPERIMENT_NAME_EFFICIENTNET")
run_name = "efficientnet_v1"
params = {
    "input_shape": INPUT_SHAPE,
    "optimizer": OPTIMIZER,
    "loss": LOSS_FUNC,
    "epochs": EPOCHS,
    "model_architecture": "EfficientNetB0"
}
tags = {
    "model_type": "EfficientNet",
    "stage": "training",
    "framework": "tensorflow"
}


model = build_efficientnet_model(input_shape=params['input_shape'])

model.compile(
    optimizer=params['optimizer'], 
    loss=params['loss'], 
    metrics=['accuracy']
    )

# Load data generators
train_gen, val_gen = get_data_generators(data_dir=DATA_DIR,
        target_size=IMAGE_SIZE,  # IMAGE_SIZE
        batch_size=BATCH_SIZE,  # BATCH_SIZE
        classes=CLASSES  # real = 0, fake = 1
        )

# Train the model
history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

# Save the model
model.save(config.get("model.MODEL_EFFICIENTNET_FULL"))  # Save full model
model.save_weights(config.get("model.MODEL_EFFICIENTNET_WEIGHT"))  # Save weights

# Get final metrics (last epoch)
metrics = {
    "val_accuracy": history.history['val_accuracy'][-1],
    "val_loss": history.history['val_loss'][-1],
    "train_accuracy": history.history['accuracy'][-1],
    "train_loss": history.history['loss'][-1],
}

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
    run_prefix="efficientNet_experiment_run",
    metrics=metrics,
    output_dir=OUTPUT_DIR  # Directory to save MLflow logs
    )