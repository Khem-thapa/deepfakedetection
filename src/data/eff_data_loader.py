from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input

def get_data_generators(data_dir='dataset/ef_openfaceforensic/', batch_size=16, target_size=(224, 224), classes=['real', 'fake']):
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
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        validation_split=0.2   # 80/20 train-val split
    )

    train_gen = train_datagen.flow_from_directory(
        data_dir,   
        target_size=target_size,
        batch_size=batch_size,
        class_mode="binary",
        subset="training",
        shuffle=True
    )

    val_gen = train_datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode="binary",
        subset="validation",
        shuffle=False
    )

    return train_gen, val_gen
