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

from models.meso.meso4 import Meso4Model
from data.eval_data_loader import get_data_generators
from utils.config_loader import ConfigLoader

# === Load configuration ===
config = ConfigLoader("src/config/config.yaml")

IMAGE_SIZE = tuple(config.get("data.IMAGE_SIZE"))
FULL_MODEL = config.get("model.MODEL_MESO4_FULL")
CLASSES = config.get("data.CLASSES")  # e.g., ['real', 'fake']
BATCH_SIZE = config.get("train.BATCH_SIZE")

base_path = os.path.abspath(os.getcwd(), 'dataset/test_openface')
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
accuracy = accuracy_score(y_true, y_preds)
precision = precision_score(y_true, y_preds)
recall = recall_score(y_true, y_preds)
f1 = f1_score(y_true, y_preds)
auc = roc_auc_score(y_true, y_probs)  # Use probabilities for AUC

import json

metrics = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "auc": auc
}

# Save metrics to JSON file
os.makedirs("results/evaluate", exist_ok=True)
with open("results/evaluate/metrics.json", "w") as f:
    json.dump(metrics, f)

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

# Save confusion matrix
plt.savefig("results/evaluate/confusion_matrix.png")
plt.show()