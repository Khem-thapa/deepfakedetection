import os
import io
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class MLFlowLogger:
    def __init__(self, output_dir="results/mlflow_logs/"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def log_training_curves(self, history):
        plt.figure(figsize=(12, 5))

        # Accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history["accuracy"], label="Train Acc")
        plt.plot(history.history["val_accuracy"], label="Val Acc")
        plt.title("Accuracy Curve")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()

        # Loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history["loss"], label="Train Loss")
        plt.plot(history.history["val_loss"], label="Val Loss")
        plt.title("Loss Curve")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()

        # Save and log
        path = os.path.join(self.output_dir, "training_curves.png")
        plt.savefig(path)
        mlflow.log_artifact(path)
        plt.close()

    def log_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)

        cm_path = os.path.join(self.output_dir, "confusion_matrix_log.png")
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path)
        plt.close()

    def log_text(self, text, filename="run_notes.txt"):
        text_path = os.path.join(self.output_dir, filename)
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(text)
        mlflow.log_artifact(text_path)

    def log_model_summary(self, model, filename="model_summary.txt"):
        summary_io = io.StringIO() # Use StringIO to capture the summary output
        model.summary(print_fn=lambda x: summary_io.write(x + "\n"))
        summary_path = os.path.join(self.output_dir, filename)
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary_io.getvalue())
        mlflow.log_artifact(summary_path)
