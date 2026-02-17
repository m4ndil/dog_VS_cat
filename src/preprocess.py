import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from .config import BASE_DATA_DIR, IMG_SIZE, BATCH_SIZE


def remove_corrupted_images():

    print("Checking for corrupted images...")

    for category in ["Cat", "Dog"]:
        folder = os.path.join(BASE_DATA_DIR, category)

        for filename in os.listdir(folder):

            file_path = os.path.join(folder, filename)

            try:
                with open(file_path, "rb") as f:
                    f.read()

            except:
                print(f"Removing corrupted file: {file_path}")
                os.remove(file_path)


def get_train_validation_generators():

    remove_corrupted_images()

    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True
    )

    train_generator = datagen.flow_from_directory(
        BASE_DATA_DIR,
        classes=["Cat", "Dog"],
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="training"
    )

    validation_generator = datagen.flow_from_directory(
        BASE_DATA_DIR,
        classes=["Cat", "Dog"],
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="validation"
    )

    return train_generator, validation_generator


def get_test_generator():

    datagen = ImageDataGenerator(rescale=1.0 / 255)

    generator = datagen.flow_from_directory(
        BASE_DATA_DIR,
        classes=["Cat", "Dog"],
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=1,
        class_mode=None,
        shuffle=False
    )

    return generator