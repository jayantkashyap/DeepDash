from PIL import Image
import numpy as np
import pickle
import sys
import cv2
import os

sys.path.append('..')
from utils.config import Config


def build_nn_dataset_generator():

    from keras.applications.vgg19 import preprocess_input
    from keras.preprocessing import image

    train_datagen = image.ImageDataGenerator(
        preprocessing_function=preprocess_input,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        fill_mode='nearest',
    )

    val_datagen = image.ImageDataGenerator(
        preprocessing_function=preprocess_input,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        fill_mode='nearest',
    )

    train_generator = train_datagen.flow_from_directory(
        f'{Config.DATASET_DIR}/{Config.ENTITY_NAME}/train',
        target_size=Config.TARGET_SIZE,
        batch_size=Config.BATCH_SIZE
    )

    validation_generator = val_datagen.flow_from_directory(
        f'{Config.DATASET_DIR}/{Config.ENTITY_NAME}/val',
        target_size=Config.TARGET_SIZE,
        batch_size=Config.BATCH_SIZE
    )

    return train_generator, validation_generator


def build_knn_dataset():
    pass


if __name__ == "__main__":
    pass
