from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data_generators(data_dir='dataset/openfaceforensic/', batch_size=16, target_size=(224, 224), classes=['real', 'fake']):
    """Create data generators for training and validation datasets.
    Args:
        data_dir (str): Directory containing the dataset.
        batch_size (int): Size of the batches of data.
        target_size (tuple): Target size for the images.
        classes (list): List of class names.
    Returns:
        train_gen: Training data generator.
        val_gen: Validation data generator.
    """

    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        validation_split=0.2 
        )

    classes = classes  # real = 0, fake = 1  

    train_gen = datagen.flow_from_directory(
        data_dir,
        target_size=target_size,  # IMAGE_SIZE
        batch_size=batch_size,  # BATCH_SIZE
        class_mode='binary',
        subset='training',
        classes=classes,
        shuffle=True
    )

    val_gen = datagen.flow_from_directory(
        data_dir,
        target_size=target_size,  # IMAGE_SIZE
        batch_size=batch_size,  # BATCH_SIZE
        class_mode='binary',
        subset='validation',
        classes=classes,
        shuffle=False
    )

    return train_gen, val_gen
