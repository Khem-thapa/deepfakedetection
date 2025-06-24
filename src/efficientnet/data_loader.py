from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data_generators():
    """    Create data generators for training and validation datasets.
    Returns:
        train_gen: Training data generator.
        val_gen: Validation data generator.
    """
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20, # Randomly rotate images
        horizontal_flip=True, # Randomly flip images horizontally
        vertical_flip=True, # Randomly flip images vertically
        zoom_range=0.2,
        validation_split=0.2
    )

    # Load images from directory and apply transformations
    train_gen = datagen.flow_from_directory(
        'dataset/fake_and_real/', 
        target_size=(256, 256), 
        batch_size=32, 
        class_mode='binary', 
        subset='training'
        )
    
    # Validation generator
    # Use the same directory but with a different subset for validation
    val_gen = datagen.flow_from_directory(
        'dataset/fake_and_real/', 
        target_size=(256, 256),
        batch_size=32, 
        class_mode='binary', 
        subset='validation' 
        )
    
    return train_gen, val_gen