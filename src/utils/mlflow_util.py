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
    train_gen=None,
    val_gen=None,
    run_prefix="df_experiment_run",
    metrics = None
    ):
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
        # mlflow.log_metric("val_accuracy", float(val_accuracy))
        if metrics:
            logger = MLFlowLogger()
            for key, value in metrics.items():
                mlflow.log_metric(key, round(value, 4))
            
        # === Log model ===
        if model and train_gen is not None and val_gen is not None:
            keras_model = model.model if hasattr(model, "model") else model # Assuming model has a 'model' attribute for Keras models
            sample_X, sample_y = next(train_gen)  # Get a small batch of data for signature inference, that is exactly one batch of data
            sample_X = np.array(sample_X)  # Ensure sample_X is a numpy array
            signature = infer_signature(sample_X, keras_model.predict(sample_X))  # Infer signature from a small sample of training data
            mlflow.keras.log_model(
                keras_model, 
                artifact_path="models", 
                signature=signature, 
                # input_example=sample_X,  # as input_example is not supported in mlflow.keras.log_model for image data
                registered_model_name=run_name)
            
        # === Log model summary, curves etc. ===
        if history or model:
            logger = MLFlowLogger()
            if history:
                logger.log_training_curves(history)
            if model:
                logger.log_model_summary(model)  

            # Log confusion matrix and ROC curve    
            if train_gen is not None and val_gen is not None:
                logger.log_confusion_matrix(model, val_gen)
                logger.log_roc_curve(model, val_gen)

            # === Log in json format ===
            if model:
                summary_data = {
                    "run_name": run_name,
                    "experiment_name": experiment_name,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "run_id": mlflow.active_run().info.run_id,
                    "params": params if params else {},
                    "tags": tags if tags else {},
                    "metrics": {k: round(v, 4) for k, v in metrics.items()} if metrics else {}
                }
                logger.log_json_summary(summary_data) 
        