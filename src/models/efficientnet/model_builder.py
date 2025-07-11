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

    efnetB0 = base_model.output # the output is the last layer of the base model
    efnetB0 = GlobalAveragePooling2D()(efnetB0) # Global average pooling to reduce dimensions
    efnetB0 = Dropout(0.5)(efnetB0) # Dropout for regularization
    efnetB0 = Dense(128, activation='relu')(efnetB0) # Fully connected layer with ReLU activation
    efnetB0 = Dropout(0.3)(efnetB0)
    output = Dense(num_classes, activation='sigmoid')(efnetB0) # Output layer for binary classification
    
    return Model(inputs=base_model.input, outputs=output)