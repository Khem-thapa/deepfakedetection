from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, Activation, AveragePooling2D
from keras.layers import Flatten, Dropout, Dense
from keras.optimizers import Adam
import numpy as np

class Meso4Model:
    def __init__(self, input_shape=(256, 256, 3), learning_rate=0.001):
        self.model = self._build_model(input_shape)
        self.model.compile(optimizer=Adam(learning_rate=learning_rate),
                           loss='binary_crossentropy',
                           metrics=['accuracy'])

    def _build_model(self, input_shape):
        model = Sequential()

        model.add(Conv2D(8, (3, 3), padding='same', input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))

        model.add(Conv2D(8, (5, 5), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))

        model.add(Conv2D(16, (5, 5), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))

        model.add(Conv2D(16, (5, 5), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(AveragePooling2D(pool_size=(4, 4), padding='same'))

        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(16))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        return model

    def train(self, x_train, y_train, x_val=None, y_val=None, batch_size=32, epochs=10):
        return self.model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val) if x_val is not None else None,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1
        )

    def evaluate(self, x_test, y_test):
        loss, accuracy = self.model.evaluate(x_test, y_test, verbose=1)
        print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
        return loss, accuracy

    def predict(self, x_input):
        predictions = self.model.predict(x_input)
        return predictions

    def save(self, path):
        if path.endswith(".weights.h5"):
            self.model.save_weights(path)  # Ensure the path ends with .weights.h5
            print(f"Model weights saved to {path}")
        else:
            self.model.save(path)  # This saves the full model
            print(f"Model saved to {path}")


    def load(self, path):
        self.model.load_weights(path)
        print(f"Model weights loaded from {path}")

    def summary(self, *args, **kwargs):
        return self.model.summary(*args, **kwargs)
