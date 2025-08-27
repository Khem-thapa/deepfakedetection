# Add the parent of "src" to Python path
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.eff_data_loader import get_data_generators
from models.efficientnet.model_builder import build_efficientnet_model
from utils.mlflow_util import start_mlflow_run_with_logging
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# === Load config ===

from utils.config_loader import ConfigLoader
config = ConfigLoader("src/config/config.yaml")

OUTPUT_DIR = config.get("output.OUTPUT_DIR_EFFICIENT")

OPTIMIZER = config.get("parameters.OPTIMIZER")  # e.g., 'adam'
LOSS_FUNC = config.get("parameters.LOSS_FUNC")  # e.g., 'binary_crossentropy'
RUN_NAME = config.get("mlflow.EFFICIENTNET_RUN_NAME")
MODEL_NAME = config.get("mlflow.EFFICIENTNET_MODEL_NAME")


def main():
    try:
        # Create output directories
        model_path = config.get("model.MODEL_EFFICIENTNET_FULL")
        weights_path = config.get("model.MODEL_EFFICIENTNET_WEIGHT")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs(os.path.dirname(weights_path), exist_ok=True)

        # Load data generators
        train_gen, val_gen = get_data_generators()
        
        model = build_efficientnet_model()

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
            ModelCheckpoint(filepath="models/efficientnet/efficient_model.keras", monitor="val_auc", save_best_only=True, mode="max")
        ]
        
        # Train the model
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=5,
            callbacks=callbacks,
            verbose=2
        )
        try:
            # Define experiment config
            EXPERIMENT_NAME = config.get("mlflow.EXPERIMENT_NAME_EFFICIENTNET")
            params = {
                "input_shape": (224, 224, 3),
                "optimizer": OPTIMIZER,
                "loss": LOSS_FUNC,
                "epochs": 5,
                "model_architecture": "EfficientNetB0"
            }
            tags = {
                "model_type": "EfficientNetB0",
                "stage": "training",
                "framework": "tensorflow"
            }
            
            # Save the model
            model.save(model_path)
            model.save_weights(weights_path)

            # Get final metrics (last epoch)
            # metrics = {
            #     "val_accuracy": float(history.history['val_accuracy'][-1]),
            #     "val_loss": float(history.history['val_loss'][-1]),
            #     "train_accuracy": float(history.history['accuracy'][-1]),
            #     "train_loss": float(history.history['loss'][-1]),
            # }

            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

            # Example: assuming you used Keras to predict
            y_probs = model.predict(val_gen, verbose=1).flatten()
            y_pred = (y_probs > 0.5).astype("int32")
            y_val = val_gen.classes  # or np.array([...]) depending on your generator

            metrics = {
                "val_accuracy": float(accuracy_score(y_val, y_pred)),
                "val_precision": float(precision_score(y_val, y_pred, zero_division=0)),
                "val_recall": float(recall_score(y_val, y_pred)),
                "val_f1_score": float(f1_score(y_val, y_pred)),
                "val_auc_score": float(roc_auc_score(y_val, y_probs)),
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