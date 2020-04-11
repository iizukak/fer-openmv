from tensorflow.keras.preprocessing.image import ImageDataGenerator


def generator():
    """
    Create Data Generator with Data Augmentation
    """
    datagen = ImageDataGenerator(featurewise_center=False,
                                 featurewise_std_normalization=False,
                                 rotation_range=20,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 horizontal_flip=True)
    return datagen
