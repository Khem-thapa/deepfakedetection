# src/data/dataloader.py

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, real_dir, fake_dir, img_size=(256, 256), test_split=0.2, seed=42):
        self.real_dir = real_dir
        self.fake_dir = fake_dir
        self.img_size = img_size
        self.test_split = test_split
        self.seed = seed

    def _load_images_from_dir(self, directory, label):
        images = []
        labels = []

        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img = cv2.imread(file_path)
                img = cv2.resize(img, self.img_size)
                img = img / 255.0  # normalize to [0, 1]
                images.append(img)
                labels.append(label)

        return images, labels

    def load_data(self):
        print("[INFO] Loading real images...")
        real_images, real_labels = self._load_images_from_dir(self.real_dir, label=0)

        print("[INFO] Loading fake images...")
        fake_images, fake_labels = self._load_images_from_dir(self.fake_dir, label=1)

        X = np.array(real_images + fake_images)
        y = np.array(real_labels + fake_labels)

        print(f"[INFO] Loaded {len(X)} images. Splitting into train/test...")

        return train_test_split(X, y, test_size=self.test_split, random_state=self.seed, stratify=y)
