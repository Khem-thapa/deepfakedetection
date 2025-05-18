from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

class Meso4Model:
    def __init__(self, input_shape=(256, 256, 3)):
        self.model = self._build_model(input_shape)

    def _build_model(self, input_shape):
        input_layer = Input(shape=input_shape)

        x = Conv2D(8, (3, 3), padding='same', activation='relu')(input_layer)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Conv2D(8, (5, 5), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Conv2D(16, (5, 5), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Conv2D(16, (5, 5), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(4, 4))(x)

        x = Flatten()(x)
        x = Dense(16, activation='relu')(x)
        output_layer = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=input_layer, outputs=output_layer)
        return model

    def compile(self):
        self.model.compile(optimizer=Adam(learning_rate=0.001),
                           loss='binary_crossentropy',
                           metrics=['accuracy'])

    def train(self, X_train, y_train, X_val, y_val, callbacks=None, epochs=10, batch_size=16):
        self.compile()
        return self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )

    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        self.model.save_weights(path)

    def load(self, path):
        self.model.load_weights(path)
