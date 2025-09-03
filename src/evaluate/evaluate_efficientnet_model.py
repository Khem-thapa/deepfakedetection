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
    classification_report,
    roc_auc_score
)

import mlflow
from mlflow.tracking import MlflowClient
import mlflow.keras
from tensorflow.keras.models import load_model

# Add the parent of "src" to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.efficientnet.model_builder import build_efficientnet_model
from data.eff_data_loader import get_data_generators
from utils.config_loader import ConfigLoader

# === Load configuration ===
config = ConfigLoader("src/config/config.yaml")

FULL_MODEL = config.get("model.MODEL_EFFICIENTNET_FULL")
CLASSES = config.get("data.CLASSES")  # e.g., ['real', 'fake']
BATCH_SIZE = config.get("train.BATCH_SIZE")
RUN_NAME = config.get("mlflow.EFFICIENTNET_RUN_NAME")
MODEL_NAME = config.get("mlflow.EFFICIENTNET_MODEL_NAME")

# === Load test data ===
base_path = os.path.abspath(os.path.join(os.getcwd(), 'dataset/test_openface'))
print("Base path for evaluation data:", base_path)

_, eval_gen = get_data_generators(
    data_dir=base_path,
    batch_size=BATCH_SIZE
)

# === Load model ===
# If FULL_MODEL is weights file, use load_weights.
# If FULL_MODEL is a saved .h5 / SavedModel, use load_model.
model = load_model(FULL_MODEL)
print("Loaded full model successfully.")

# === Predict ===
y_probs = model.predict(eval_gen, verbose=1)
y_preds = (y_probs > 0.5).astype("int32").flatten()
y_true = eval_gen.classes

# === Metrics ===
metrics = {
    "accuracy": float(accuracy_score(y_true, y_preds)),
    "precision": float(precision_score(y_true, y_preds)),
    "recall": float(recall_score(y_true, y_preds)),
    "f1_score": float(f1_score(y_true, y_preds)),
    "auc": float(roc_auc_score(y_true, y_probs))
}
metrics = {k: round(v, 4) for k, v in metrics.items()}

print("Evaluation Metrics:")
for k, v in metrics.items():
    print(f"{k}: {v}")

# Save metrics
os.makedirs("results/evaluate/efficientnet", exist_ok=True)
with open("results/evaluate/efficientnet/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# === Classification Report ===
print("\nClassification Report:")
print(classification_report(y_true, y_preds, target_names=CLASSES))

# === Confusion Matrix ===
cm = confusion_matrix(y_true, y_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("results/evaluate/efficientnet/confusion_matrix.png")
plt.close()

# === MLflow Logging ===
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("eval_efficientnet_model")

try:
    with mlflow.start_run(run_name=f"evaluation_{RUN_NAME}") as run:
        print(f"Started MLflow run: {run.info.run_id}")

        # Log metrics
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        # Log artifacts
        mlflow.log_artifact("results/evaluate/efficientnet/metrics.json")
        mlflow.log_artifact("results/evaluate/efficientnet/confusion_matrix.png")

        # === Log model ===
        try:
            # Prepare input for signature
            sample_batch = next(iter(eval_gen))
            sample_input = np.array(sample_batch[0][:1])
            sample_prediction = model.predict(sample_input)

            from mlflow.models.signature import infer_signature
            signature = infer_signature(sample_input, sample_prediction)

            mlflow.keras.log_model(
                model,
                artifact_path="model",
                signature=signature,
                registered_model_name="Eval_efficientnet_Model"
            )
            print("Model logged successfully.")

            # Optional: Promote model
            # safe_model_promotion(MODEL_NAME, metrics, run.info.run_id, "model")

        except Exception as model_error:
            print(f"Error logging model: {model_error}")

except Exception as e:
    print(f"MLflow logging failed: {e}")

print("Evaluation complete!")
