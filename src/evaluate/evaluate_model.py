import sys
import os
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

# Add the parent of "src" to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.meso4 import Meso4Model
from data.dataloader import DataLoader
from utils.config_loader import ConfigLoader

# === Load configuration ===
config = ConfigLoader("src/config/config.yaml")

REAL_DIR = config.get("data.REAL_DIR")
FAKE_DIR = config.get("data.FAKE_DIR")
IMAGE_SIZE = tuple(config.get("data.IMAGE_SIZE"))
FULL_MODEL = config.get("output.MODEL_FULL_PATH")

# === Load test data ===
loader = DataLoader(real_dir=REAL_DIR, fake_dir=FAKE_DIR)
_, X_test, _, y_test = loader.load_data()

# === Load model and weights ===
model = Meso4Model()
model.load(FULL_MODEL)

# === Predict ===
y_probs = model.predict(X_test)
y_pred = (y_probs > 0.5).astype("int32").flatten()

# === Metrics ===
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_probs)

import json

metrics = {
    "accuracy": acc,
    "precision": prec,
    "recall": rec,
    "f1_score": f1,
    "auc": auc
}

# Save metrics to JSON file
os.makedirs("results/evaluate", exist_ok=True)
with open("results/evaluate/metrics.json", "w") as f:
    json.dump(metrics, f)

# === Classification Report ===
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))

# === Confusion Matrix ===
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()

# Save confusion matrix
plt.savefig("results/evaluate/confusion_matrix.png")
plt.show()
