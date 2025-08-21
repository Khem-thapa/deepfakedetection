import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report, roc_auc_score
)

import mlflow
from mlflow.tracking import MlflowClient
import mlflow.keras

# Add the parent of "src" to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.meso.meso4 import Meso4Model
from data.eval_data_loader import get_data_generators
from utils.config_loader import ConfigLoader

# === Load configuration ===
config = ConfigLoader("src/config/config.yaml")

IMAGE_SIZE = tuple(config.get("data.IMAGE_SIZE"))
FULL_MODEL = config.get("model.MODEL_MESO4_FULL")
CLASSES = config.get("data.CLASSES")  # e.g., ['real', 'fake']
BATCH_SIZE = config.get("train.BATCH_SIZE")
RUN_NAME = config.get("mlflow.MESO4_RUN_NAME")
MODEL_NAME = config.get("mlflow.MESO4_MODEL_NAME")

base_path = os.path.abspath(os.path.join(os.getcwd(), 'dataset/test_openface'))
print("Base path for evaluation data:", base_path)
# === Load test data ===
eval_gen = get_data_generators(
    data_dir= base_path,
    target_size=IMAGE_SIZE,  # IMAGE_SIZE
    batch_size=BATCH_SIZE,  # BATCH_SIZE
    classes=CLASSES  # real = 0, fake = 1
)



# === Load model and weights ===
model = Meso4Model()
model.load(FULL_MODEL)

# === Predict ===
y_probs = model.predict(eval_gen, verbose=1)
y_preds = (y_probs > 0.5).astype("int32").flatten()

# True classes
y_true = eval_gen.classes

# === Metrics ===
metrics = {
    "accuracy": float(accuracy_score(y_true, y_preds)),
    "precision": float(precision_score(y_true, y_preds)),
    "recall": float(recall_score(y_true, y_preds)),
    "f1_score": float(f1_score(y_true, y_preds)),
    "auc": float(roc_auc_score(y_true, y_probs))
}

# Round metrics to avoid precision issues
metrics = {k: round(v, 4) for k, v in metrics.items()}

print("Evaluation Metrics:")
for key, value in metrics.items():
    print(f"{key}: {value}")

# Save metrics to JSON
os.makedirs("results/evaluate", exist_ok=True)
with open("results/evaluate/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# === Classification Report ===
print("\nClassification Report:")
print(classification_report(y_true, y_preds, target_names=["Real", "Fake"]))

# === Confusion Matrix ===
cm = confusion_matrix(y_true, y_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("results/evaluate/confusion_matrix.png")
plt.close()

# === MLflow Logging & Promotion ===
def safe_model_promotion(model_name, metrics, run_id, model_artifact):
    """Safely handle model promotion with error handling"""
    try:
        client = MlflowClient()
        
        # Check current Production model
        prod_versions = client.get_latest_versions(model_name, stages=["Production"])
        prod_acc = 0.0
        
        if prod_versions:
            prod_run = client.get_run(prod_versions[0].run_id)
            prod_metrics = prod_run.data.metrics
            prod_acc = prod_metrics.get("accuracy", 0.0)
            
            # Ensure it's a float, not a Metric object
            if hasattr(prod_acc, 'value'):
                prod_acc = float(prod_acc.value)
            else:
                prod_acc = float(prod_acc)

        current_acc = metrics["accuracy"]
        print(f"Current model accuracy: {current_acc}")
        print(f"Production model accuracy: {prod_acc}")

        # Compare and promote
        if current_acc > prod_acc:
            print(f"Promoting new model ({current_acc:.4f}) > Production ({prod_acc:.4f})")
            
            # Register model
            model_uri = f"runs:/{run_id}/{model_artifact}"
            reg_model = mlflow.register_model(model_uri, model_name)
            
            # Transition to Production
            client.transition_model_version_stage(
                name=model_name,
                version=str(reg_model.version),  # Ensure version is string
                stage="Production",
                archive_existing_versions=True
            )
            print(f"Successfully promoted model version {reg_model.version} to Production")
            return True
        else:
            print(f"New model ({current_acc:.4f}) <= Production ({prod_acc:.4f}), not promoting.")
            return False
            
    except Exception as e:
        print(f"Error during model promotion: {e}")
        
        # Try first-time registration
        try:
            print("Attempting first-time model registration...")
            model_uri = f"runs:/{run_id}/{model_artifact}"
            reg_model = mlflow.register_model(model_uri, model_name)
            
            client.transition_model_version_stage(
                name=model_name,
                version=str(reg_model.version),
                stage="Production"
            )
            print(f"Successfully registered and promoted first model version {reg_model.version}")
            return True
            
        except Exception as e2:
            print(f"Failed to register model: {e2}")
            print("Skipping model promotion due to errors.")
            return False

# === MLflow Logging ===
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("eval_meso4_model")

try:
    with mlflow.start_run(run_name=f"evaluation_{RUN_NAME}") as run:
        print(f"Started MLflow run: {run.info.run_id}")
        
        # Log metrics
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
        
        # Log artifacts
        mlflow.log_artifact("results/evaluate/metrics.json") # Metrics
        mlflow.log_artifact("results/evaluate/confusion_matrix.png") # Confusion Matrix

        # Log model with updated parameter name
        try:
            # Create a sample input for signature inference
            sample_batch = next(iter(eval_gen))
            sample_input = np.array(sample_batch[0][:1])  # Take first sample
            
            # Get model prediction for signature
            sample_prediction = model.model.predict(sample_input)
            
            # Infer signature
            from mlflow.models.signature import infer_signature
            signature = infer_signature(sample_input, sample_prediction)
            
            # Log model with signature
            mlflow.keras.log_model(
                model.model, 
                name="model",  # Keep this for now to avoid issues
                signature=signature,
                registered_model_name="Eval_Meso4_Model",
            )
            print("Model logged successfully")
            
            # Attempt model promotion
            # safe_model_promotion(MODEL_NAME, metrics, run.info.run_id, "model")
            
        except Exception as model_error:
            print(f"Error logging model: {model_error}")
        
except Exception as e:
    print(f"MLflow logging failed: {e}")

print("Evaluation complete!")