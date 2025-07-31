import numpy as np

def preprocess_image(image, target_size=(256, 256)):
    image = image.resize(target_size)
    arr = np.asarray(image).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)  # shape (1, 256, 256, 3)
    return arr
