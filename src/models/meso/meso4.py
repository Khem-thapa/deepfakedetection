from gc import callbacks
from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, Activation, AveragePooling2D
from keras.layers import Dropout, Dense, Flatten
from keras.optimizers import Adam
from keras.regularizers import l2
import tensorflow as tf
import numpy as np

# Meso4Model: A Convolutional Neural Network for image classification
# This model is designed to classify images into two categories (e.g., real vs fake).
class Meso4Model:
    def __init__(self, input_shape=(256, 256, 3), learning_rate=0.001, dropout_rate=0.3):
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.model = self._build_model(input_shape)
        self.model.compile(optimizer=Adam(learning_rate=self.learning_rate),
                           loss='binary_crossentropy',
                           metrics=['accuracy', 
                                    tf.keras.metrics.AUC(name='auc'), 
                                    tf.keras.metrics.Precision(name='precision'), 
                                    tf.keras.metrics.Recall(name='recall')
                                    ])

    def _build_model(self, input_shape):
        # Build the Meso4 model architecture
        # input_shape: shape of the input images (height, width, channels)
        model = Sequential()

        # Adding layers to the model    
        model.add(Conv2D(8, (3, 3), padding='same', input_shape=input_shape))
        model.add(BatchNormalization()) 
        model.add(Activation('relu'))
        model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))

        # Additional convolutional layers with batch normalization and activation
        model.add(Conv2D(8, (5, 5), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))

        model.add(Conv2D(16, (5, 5), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))

        # Additional convolutional layers with batch normalization and activation
        model.add(Conv2D(16, (5, 5), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(AveragePooling2D(pool_size=(4, 4), padding='same'))

        # Flatten the output from the convolutional layers
        # and add fully connected layers
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(16))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        return model

    # Train the model
    # x_train: training data, y_train: training labels
    def train(self, train_gen, val_gen, batch_size=32, epochs=10, **kwargs):
        return self.model.fit(
            train_gen,
            validation_data= val_gen,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            **kwargs
        )

    # Evaluate the model
    # x_test: test data, y_test: test labels
    def evaluate(self, x_test, y_test):
        loss, accuracy = self.model.evaluate(x_test, y_test, verbose=1)
        print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
        return loss, accuracy

    # Predict using the model
    # x_input: input data for prediction
    def predict(self, x_input, **kwargs):
        predictions = self.model.predict(x_input, **kwargs)
        return predictions

    # Save the model or its weights
    def save(self, path):
        if path.endswith(".weights.h5"):
            self.model.save_weights(path)  # Ensure the path ends with .weights.h5
            print(f"Model weights saved to {path}")
        else:
            self.model.save(path)  # This saves the full model
            print(f"Model saved to {path}")


    # Load model weights or the full model
    def load(self, path):
        if path.endswith(".weights.h5"):
            self.model.load_weights(path)  # Ensure the path ends with .weights.h5
            print(f"Model weights loaded from {path}")  
        else:
            # If the path does not end with .weights.h5, assume it's a full model
            from keras.models import load_model
            self.model = load_model(path)
            print(f"Model loaded from {path}")
            return 
        
    # Get model summary
    def summary(self, *args, **kwargs):
        return self.model.summary(*args, **kwargs)
