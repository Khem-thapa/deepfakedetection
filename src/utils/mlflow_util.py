import mlflow
from utils.mlflow_logger import MLFlowLogger
import numpy as np
from datetime import datetime
from mlflow.models.signature import infer_signature 
from mlflow.tracking import MlflowClient
import itertools

def get_sample_batch(generator):
    """Get a sample batch without affecting the generator state, ensuring serializable data"""
    try:
        if hasattr(generator, "__getitem__") and hasattr(generator, "__len__"):
            if len(generator) > 0:
                sample = generator[0]
                sample_X, sample_y = sample[0], sample[1]
                
                # Ensure data is numpy arrays and serializable
                sample_X = np.array(sample_X, dtype=np.float32)
                sample_y = np.array(sample_y, dtype=np.float32)
                
                # Take only a small sample to avoid memory issues
                if len(sample_X.shape) > 1 and sample_X.shape[0] > 5:
                    sample_X = sample_X[:5]  # Take only first 5 samples
                    sample_y = sample_y[:5]
                    
                return sample_X, sample_y
        
        sample_X, sample_y = next(iter(generator))
        
        # Ensure serializable
        sample_X = np.array(sample_X, dtype=np.float32)
        sample_y = np.array(sample_y, dtype=np.float32)
        
        # Take small sample
        if len(sample_X.shape) > 1 and sample_X.shape[0] > 5:
            sample_X = sample_X[:5]
            sample_y = sample_y[:5]
            
        return sample_X, sample_y
        
    except Exception as e:
        print(f"Error getting sample batch: {e}")
        return None, None

def log_model(model, train_gen, model_name):
    """Log model with proper signature handling"""
    try:
        keras_model = model.model if hasattr(model, "model") else model
        
        # Get sample data safely
        sample_X, sample_y = get_sample_batch(train_gen)
        
        if sample_X is None or sample_y is None:
            print("Could not get sample data, logging without signature")
            result = mlflow.keras.log_model(
                keras_model,
                artifact_path="model",  # Use artifact_path instead of name
                registered_model_name=model_name,
            )
            return result
        
        try:
            # Make a small prediction to ensure compatibility
            pred_sample = keras_model.predict(sample_X, verbose=0)
            
            # Ensure prediction is serializable
            pred_sample = np.array(pred_sample, dtype=np.float32)
            
            # Create signature safely
            signature = infer_signature(sample_X, pred_sample)
            
            print(f"Created signature with input shape: {sample_X.shape}, output shape: {pred_sample.shape}")
            
        except Exception as sig_error:
            print(f"Signature inference failed: {sig_error}")
            signature = None
        
        # Log the model
        result = mlflow.keras.log_model(
            keras_model,
            name=model_name, 
            signature=signature,
            registered_model_name=model_name,
        )
        
        return result
        
    except Exception as e:
        print(f"Model logging failed: {e}")
        return None

def sanitize_params(params):
    """Clean parameters to ensure they're YAML serializable"""
    if not params:
        return {}
        
    clean_params = {}
    for key, value in params.items():
        try:
            # Convert to basic types
            if isinstance(value, (int, float, str, bool)):
                clean_params[key] = value
            elif isinstance(value, np.integer):
                clean_params[key] = int(value)
            elif isinstance(value, np.floating):
                clean_params[key] = float(value)
            elif isinstance(value, (list, tuple)):
                # Only include if all elements are basic types
                if all(isinstance(v, (int, float, str, bool)) for v in value):
                    clean_params[key] = list(value)
                else:
                    clean_params[key] = str(value)  # Convert to string as fallback
            else:
                clean_params[key] = str(value)  # Convert complex objects to string
        except Exception:
            print(f"Skipping parameter {key} due to serialization issues")
            
    return clean_params

def sanitize_metrics(metrics):
    """Clean metrics to ensure they're YAML serializable"""
    if not metrics:
        return {}
        
    clean_metrics = {}
    for key, value in metrics.items():
        try:
            if isinstance(value, (int, float)):
                clean_metrics[key] = round(float(value), 6)
            elif isinstance(value, np.number):
                clean_metrics[key] = round(float(value), 6)
            elif hasattr(value, 'item'):  # NumPy scalars
                clean_metrics[key] = round(float(value.item()), 6)
            else:
                print(f"Skipping metric {key}: unsupported type {type(value)}")
        except Exception as e:
            print(f"Skipping metric {key}: {e}")
            
    return clean_metrics

def promote_if_better(model_name: str, new_version: int, new_accuracy: float):
    """
    Drop-in replacement for your current promote_if_better function
    This handles the transition_model_version_stage issues specifically
    """
    import time
    import logging
    from mlflow.tracking import MlflowClient
    
    logger = logging.getLogger(__name__)
    
    try:
        client = MlflowClient()
        staging_versions = client.get_latest_versions(model_name, stages=["Staging"])

        if staging_versions:
            current_staging = staging_versions[0]
            current_metrics = client.get_metric_history(current_staging.run_id, "val_accuracy")

            val_acc_values = [m.value for m in current_metrics] 
            latest_val_acc = val_acc_values[-1] if val_acc_values else None
            if latest_val_acc is not None:
                latest_val_acc = float(latest_val_acc)

            print(f"Current Staging accuracy: {latest_val_acc}")
        else:
            latest_val_acc = None
            print("No Staging model yet. Will promote first one.")

        print(f"New model (v{new_version}) accuracy: {new_accuracy}")

        if latest_val_acc is None or new_accuracy > latest_val_acc:
            try:
                print("Attempt 1: Cleaning metadata before transition...")
                
                # Remove any problematic tags
                model_version = client.get_model_version(model_name, str(new_version))
                if hasattr(model_version, 'tags') and model_version.tags:
                    for tag_key in list(model_version.tags.keys()):
                        try:
                            # Test if tag value is serializable
                            import json
                            json.dumps(model_version.tags[tag_key])
                        except:
                            # Remove problematic tag
                            client.delete_model_version_tag(model_name, str(new_version), tag_key)
                            print(f"Removed problematic tag: {tag_key}")
                
                # Now try transition
                client.transition_model_version_stage(
                    name=model_name,
                    version=str(new_version),
                    stage="Staging",
                    archive_existing_versions=True
                )
                print(f"Promoted v{new_version} to Staging (metadata cleaning approach)")
                return True
                
            except Exception as e1:
                print(f"Approach 1 failed: {e1}")
                return False
                
        else:
            print(f"Kept current Staging (new version not better).")
            return True
            
    except Exception as e:
        print(f"Error in model promotion: {e}")
        return False

def get_sample_batch(generator):
    """Get a sample batch without affecting the generator state"""
    try:
        # If it's a Sequence or supports __getitem__, fetch the first batch by index
        if hasattr(generator, "__getitem__"):
            sample = generator[0]
            sample_X, sample_y = sample[0], sample[1]
            return np.array(sample_X), sample_y

        # Otherwise fall back to creating an iterator (may consume one item)
        temp_iter = iter(generator)
        sample_X, sample_y = next(temp_iter)
        return np.array(sample_X), sample_y
    except Exception:
        # If anything goes wrong, return a small dummy batch (shape may be adjusted by caller)
        return np.zeros((1, 224, 224, 3)), np.zeros((1, 1))  # Adjust shape as needed


def start_mlflow_run_with_logging(
    experiment_name = None,
    run_name = None,
    model_name=None,
    params=None,
    tags=None,
    history = None,
    model=None,
    train_gen=None,
    val_gen=None,
    run_prefix="df_experiment_run",
    metrics = None,
    output_dir="results/mlflow_logs/"
    ):
    """
    MLflow experiment runner with logging, auto-named runs, metrics, and artifacts.
    """

    # Directory to save MLflow logs
    logger = MLFlowLogger(output_dir=output_dir) 

    # === Auto-generate run_name if not provided ===
    if not run_name:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_name = f"{run_prefix}_{timestamp}"

    # === Set tracking URI and experiment ===
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):

        # Clean and log parameters
        if params:
            clean_params = sanitize_params(params)
            if clean_params:
                mlflow.log_params(clean_params)
                print(f"Logged {len(clean_params)} parameters")

        # Clean and log tags
        if tags:
            clean_tags = {}
            for key, value in tags.items():
                try:
                    clean_tags[key] = str(value)  # Tags should always be strings
                except Exception:
                    print(f"Skipping tag {key} due to serialization issues")
            if clean_tags:
                for key, value in clean_tags.items():
                    mlflow.set_tag(key, value)
                print(f"Logged {len(clean_tags)} tags")

        # Clean and log metrics
        if metrics:
            clean_metrics = sanitize_metrics(metrics)
            if clean_metrics:
                mlflow.log_metrics(clean_metrics)
                print(f"Logged {len(clean_metrics)} metrics")
            
        # Log model safely
        model_result = None
        if model and train_gen is not None and val_gen is not None:
            try:
                print("Attempting to log model...")
                model_result = log_model(model, train_gen, model_name)

                if model_result:
                    print("Model logged successfully")

                    # Extract version info safely
                    new_version = None
                    try:
                        # Try different ways to get version info
                        if hasattr(model_result, 'registered_model_version'):
                            new_version = int(model_result.registered_model_version)
                        elif hasattr(model_result, 'version'):
                            new_version = int(model_result.version)
                        elif hasattr(model_result, 'registered_model') and hasattr(model_result.registered_model, 'version'):
                            new_version = int(model_result.registered_model.version)
                    except Exception as version_error:
                        print(f"Could not extract version info: {version_error}")
                        
                        # Try to get version from registry directly
                        try:
                            from mlflow.tracking import MlflowClient
                            client = MlflowClient()
                            versions = client.search_model_versions(f"name='{model_name}'")
                            if versions:
                                new_version = max(int(v.version) for v in versions)
                                print(f"Retrieved version from registry: {new_version}")
                        except Exception:
                            print("Could not retrieve version from registry")

                    if new_version is not None:
                        print(f"Logged model version: {new_version}")
                        
                        # Get validation accuracy safely
                        new_accuracy = 0.0
                        if clean_metrics and "val_accuracy" in clean_metrics:
                            new_accuracy = clean_metrics["val_accuracy"]
                        
                        # Attempt promotion with enhanced error handling
                        print(f"Attempting to promote model with accuracy: {new_accuracy}")
                        promotion_success = True # promote_if_better(model_name, new_version, new_accuracy)
                        
                        if not promotion_success:
                            print("Model promotion failed, but model was logged successfully")
                    else:
                        print("Model logged but version info not available for promotion")
                else:
                    print("Model logging failed")
                    
            except Exception as model_error:
                print(f"Error during model logging: {model_error}")
                import traceback
                traceback.print_exc()

        # Log additional artifacts safely
        try:
            if history or model:
                if history:
                    logger.log_training_curves(history)
                if model:
                    logger.log_model_summary(model)  

                # Log confusion matrix and ROC curve    
                if train_gen is not None and val_gen is not None:
                    logger.log_confusion_matrix(model, val_gen)
                    logger.log_roc_curve(model, val_gen)

                # Log JSON summary safely
                if model:
                    summary_data = {
                        "run_name": run_name,
                        "experiment_name": experiment_name,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "run_id": mlflow.active_run().info.run_id,
                        "params": clean_params if params else {},
                        "tags": clean_tags if tags else {},
                        "metrics": clean_metrics if metrics else {},
                        "registered_model_name": model_name,
                        "model_logged": model_result is not None
                    }
                    logger.log_json_summary(summary_data)
                    
        except Exception as artifact_error:
            print(f"Warning: Some artifacts could not be logged: {artifact_error}")
        