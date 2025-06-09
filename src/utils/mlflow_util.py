import mlflow
from utils.mlflow_logger import MLFlowLogger
import numpy as np
from datetime import datetime
from mlflow.models.signature import infer_signature 

def start_mlflow_run_with_logging(
    experiment_name = None,
    run_name = None,
    params=None,
    tags=None,
    history = None,
    model=None,
    val_accuracy=0.0,
    run_prefix="df_experiment_run",
    X_train=None,
    y_train=None):
    """
    MLflow experiment runner with logging, auto-named runs, metrics, and artifacts.
    """

    # === Auto-generate run_name if not provided ===
    if not run_name:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_name = f"{run_prefix}_{timestamp}"

    # === Set tracking URI and experiment ===
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):

        # === Log parameters ===
        if params:
            mlflow.log_params(params)

        # === Log tags ===
        if tags:
            for key, value in tags.items():
                mlflow.set_tag(key, str(value))

        # === Log metrics ===
        mlflow.log_metric("val_accuracy", float(val_accuracy))

        # === Log model ===
        if model and X_train is not None and y_train is not None:
            keras_model = model.model if hasattr(model, "model") else model # Assuming model has a 'model' attribute for Keras models
            sample_X = X_train[:10] if len(X_train) > 10 else X_train
            signature = infer_signature(sample_X, keras_model.predict(sample_X))  # Infer signature from a small sample of training data
            mlflow.keras.log_model(
                keras_model, 
                artifact_path="models", 
                signature=signature, 
                # input_example=sample_X,  # as input_example is not supported in mlflow.keras.log_model and these are image data
                registered_model_name=run_name)
            
        # === Log model summary, curves etc. ===
        if history or model:
            logger = MLFlowLogger()
            if history:
                logger.log_training_curves(history)
            if model:
                logger.log_model_summary(model)

            logger.log_text(f"Completed run with val_accuracy: {val_accuracy:.4f}")
