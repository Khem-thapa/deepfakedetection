from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input

def get_data_generators(data_dir, batch_size, target_size=(224, 224), classes=['real', 'fake']):
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

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.7, 1.2],
        validation_split=0.2,   # 80/20 train-val split
        fill_mode='nearest',
        zoom_range=0.2,
        shear_range=0.2
    )

    train_gen = train_datagen.flow_from_directory(
        data_dir,   
        target_size=target_size,
        batch_size=batch_size,
        class_mode="binary",
        subset="training",
        classes=classes,
        shuffle=True
    )

    val_gen = train_datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode="binary",
        subset="validation",
        classes=classes,
        shuffle=False
    )

    return train_gen, val_gen
