# src/trainer/train_model.py

import os
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, model, X_train, y_train, X_val, y_val, X_test=None, y_test=None,  save_dir='models/', batch_size=32, epochs=10):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.batch_size = batch_size
        self.epochs = epochs
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def train(self):
        checkpoint_path = os.path.join(self.save_dir, 'meso4.weights.h5')

        checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy', # Tracking validation accuracy
            save_best_only=True,
            save_weights_only=True, # save only the model's learned weights, # must end in .weights.h5 if save_weights_only=True 
            verbose=1,
            mode='max' # save the model with the highest validation accuracy, mode = min when tracking loss
        )

        history = self.model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[checkpoint],
            verbose=2
        )

        print(f"[INFO] Training complete. Best model saved to {checkpoint_path}")
        return history

    def evaluate(self, outpath):
        print("[INFO] Evaluating model on validation set...")
        preds_prob = self.model.predict(self.X_val)
        preds = (preds_prob > 0.5).astype(int).flatten()

        print(classification_report(self.y_val, preds, target_names=["Real", "Fake"]))

        # Confusion Matrix
        cm = confusion_matrix(self.y_val, preds)

        # Plot and save
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake']) # suppress warnings related to zero division
        
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')

        os.makedirs('results', exist_ok=True)
        plt.savefig(outpath)
        plt.close()

        print(f"Confusion matrix saved to {outpath}")
