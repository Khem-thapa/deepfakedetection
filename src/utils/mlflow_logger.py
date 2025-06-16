import os
import io
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc, confusion_matrix

class MLFlowLogger:
    def __init__(self, output_dir="results/mlflow_logs/"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def log_training_curves(self, history, filename="training_curves.png"):
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
        path = os.path.join(self.output_dir, filename )
        plt.savefig(path)
        mlflow.log_artifact(path)
        plt.close()

    def log_confusion_matrix(self, model, X, y_true, filename="confusion_matrix_log.png"):
        
        y_probs = model.predict(X)
        y_pred = (y_probs > 0.5).astype("int32").flatten()

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        cm_path = os.path.join(self.output_dir, filename)
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
    
    def log_roc_curve(self, model, X, y_true, filename="roc_curve.png"):
        """        Logs the ROC curve for a binary classification model.
        Args:
            model: The trained model to evaluate.
            X: The input features for prediction.
            y_true: The true labels for the input features.
            filename: The name of the file to save the ROC curve plot.
        """
        y_probs = model.predict(X)  # Get predicted probabilities
        y_pred = (y_probs > 0.5).astype("int32").flatten()
        # y_prob = model.predict(X).ravel()  # ravel to flatten the array if needed, it converts a multi-dimensional array to a 1D array
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        auc_score = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")

        roc_path = os.path.join(self.output_dir, filename)
        plt.savefig(roc_path)
        mlflow.log_artifact(roc_path)
        plt.close()

    def log_json_summary(self, summary_data, filename="run_summary.json"):
        import json
        summary_path = os.path.join(self.output_dir, filename)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=4)
        mlflow.log_artifact(summary_path)
