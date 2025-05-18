import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

import sys
import os
# Add the parent of "src" to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.meso4 import Meso4Model
from data.dataloader import DataLoader
from utils.config_loader import ConfigLoader 

# === Load config ===
config = ConfigLoader("src/config/config.yaml")

REAL_DIR = config.get("data.REAL_DIR")
FAKE_DIR = config.get("data.FAKE_DIR")
IMAGE_SIZE = tuple(config.get("data.IMAGE_SIZE"))

CUSTOM_WEIGHTS = config.get("model.CUSTOM_WEIGHTS")
PRETRAINED_WEIGHTS = config.get("model.PRETRAINED_WEIGHTS")
RESULTS_DIR = config.get("output.RESULT_DIR")

# === Load data ===
loader = DataLoader(REAL_DIR, FAKE_DIR)
X_train, X_val, y_train, y_val = loader.load_data()

# === Load and Predict with Models ===
custom_model = Meso4Model()
custom_model.load(CUSTOM_WEIGHTS)
y_pred_custom = (custom_model.predict(X_val) > 0.5).astype("int32")

pretrained_model = Meso4Model()
pretrained_model.load(PRETRAINED_WEIGHTS)
y_pred_pretrained = (pretrained_model.predict(X_val) > 0.5).astype("int32")

# === Evaluate and Compare ===
print("[Custom Model] Classification Report:")
print(classification_report(y_val, y_pred_custom, target_names=["Real", "Fake"], zero_division=0))

print("\n [Pretrained Model] Classification Report:")
print(classification_report(y_val, y_pred_pretrained, target_names=["Real", "Fake"], zero_division=0))

# === Confusion Matrices ===
cm_custom = confusion_matrix(y_val, y_pred_custom)
cm_pretrained = confusion_matrix(y_val, y_pred_pretrained)

os.makedirs(RESULTS_DIR, exist_ok=True)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.heatmap(cm_custom, annot=True, fmt="d", cmap="Blues", xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
plt.title("Custom Model")
plt.xlabel("Predicted")
plt.ylabel("True")

plt.subplot(1, 2, 2)
sns.heatmap(cm_pretrained, annot=True, fmt="d", cmap="Greens", xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
plt.title("Pretrained Model")
plt.xlabel("Predicted")
plt.ylabel("True")

plt.tight_layout()
plot_path = os.path.join(RESULTS_DIR, "comparison_confusion_matrix.png")
plt.savefig(plot_path)
plt.close()

print(f"Confusion matrix saved to: {plot_path}")
