from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

def build_efficientnet_model(input_shape=(224, 224, 3), num_classes=1):
    """Builds an EfficientNetB0 model with a custom head for binary classification.
    Args:
        input_shape (tuple): Shape of the input images.
        num_classes (int): Number of output classes.
        trainable_layers_ratio (float): Ratio of layers to be trainable in the base model.
    Returns:
        Model: A Keras Model instance with the EfficientNetB0 architecture.
    """
    
    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_tensor=Input(shape=input_shape)
    )
    
    # Unfreeze all layers for fine-tuning
    for layer in base_model.layers:
        layer.trainable = True
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation="sigmoid")(x)
    
    model = Model(inputs=base_model.input, outputs=output)
    
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    
    return model