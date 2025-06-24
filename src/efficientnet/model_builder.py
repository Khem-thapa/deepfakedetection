from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.models import Model

def build_efficientnet_model(input_shape=(256, 256, 3), num_classes=1, trainable_layers_ratio=0.3):
    """Builds an EfficientNetB0 model with a custom head for binary classification.
    Args:
        input_shape (tuple): Shape of the input images.
        num_classes (int): Number of output classes.
        trainable_layers_ratio (float): Ratio of layers to be trainable in the base model.
    Returns:
        Model: A Keras Model instance with the EfficientNetB0 architecture.
    """
    
    # Load the EfficientNetB0 model with pre-trained weights
    base_model = EfficientNetB0(
        include_top=False, 
        weights='imagenet', 
        input_tensor=Input(shape=input_shape)
        )
    
    total_layers = len(base_model.layers)
    
    for layer in base_model.layers[:int(total_layers * (1 - trainable_layers_ratio))]:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x) # Global average pooling to reduce dimensions
    x = Dropout(0.5)(x) # Dropout for regularization
    x = Dense(128, activation='relu')(x) # Fully connected layer with ReLU activation
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation='sigmoid')(x) # Output layer for binary classification
    
    return Model(inputs=base_model.input, outputs=output)