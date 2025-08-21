# Add the parent of "src" to Python path
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.data_loader import get_data_generators
from models.efficientnet.model_builder import build_efficientnet_model
from utils.mlflow_util import start_mlflow_run_with_logging


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
RUN_NAME = config.get("mlflow.EFFICIENTNET_RUN_NAME")
MODEL_NAME = config.get("mlflow.EFFICIENTNET_MODEL_NAME")

# Define experiment config
EXPERIMENT_NAME = config.get("mlflow.EXPERIMENT_NAME_EFFICIENTNET")
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

def main():
    try:
        # Create output directories
        model_path = config.get("model.MODEL_EFFICIENTNET_FULL")
        weights_path = config.get("model.MODEL_EFFICIENTNET_WEIGHT")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs(os.path.dirname(weights_path), exist_ok=True)

        model = build_efficientnet_model(input_shape=params['input_shape'])
        model.compile(
            optimizer=params['optimizer'], 
            loss=params['loss'], 
            metrics=['accuracy']
        )

        # Load data generators
        train_gen, val_gen = get_data_generators(
            data_dir=DATA_DIR,
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            classes=CLASSES
        )

        try:
            # Train the model
            history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

            # Save the model
            model.save(model_path)
            model.save_weights(weights_path)

            # Get final metrics (last epoch)
            metrics = {
                "val_accuracy": float(history.history['val_accuracy'][-1]),
                "val_loss": float(history.history['val_loss'][-1]),
                "train_accuracy": float(history.history['accuracy'][-1]),
                "train_loss": float(history.history['loss'][-1]),
            }

            # MLflow run
            start_mlflow_run_with_logging( 
                experiment_name=EXPERIMENT_NAME, 
                run_name=RUN_NAME, 
                model_name=MODEL_NAME,
                params=params, 
                tags=tags, 
                history=history, 
                model=model, 
                train_gen=train_gen,
                val_gen=val_gen,
                run_prefix="efficientNet_experiment_run",
                metrics=metrics,
                output_dir=OUTPUT_DIR
            )

        except Exception as e:
            print(f"Error during training or saving: {e}")
            raise

    except Exception as e:
        print(f"Error in experiment: {e}")
        sys.exit(1)
    finally:
        # Cleanup
        try:
            train_gen.reset()
            val_gen.reset()
        except:
            pass

if __name__ == "__main__":
    main()