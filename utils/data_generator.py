from utils.config import Config
from PIL import Image
from tqdm import tqdm
import numpy as np
import pickle
import sys
import cv2
import os

sys.path.append('..')


def build_nn_dataset_generator_VGG19():

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
        # f'{Config.DATASET_DIR}/{Config.ENTITY_NAME}/train',
        os.path.join(Config.DATASET_DIR, Config.ENTITY_NAME, 'train'),
        target_size=Config.TARGET_SIZE,
        batch_size=Config.BATCH_SIZE
    )

    validation_generator = val_datagen.flow_from_directory(
        # f'{Config.DATASET_DIR}/{Config.ENTITY_NAME}/val',
        os.path.join(Config.DATASET_DIR, Config.ENTITY_NAME, 'test'),
        target_size=Config.TARGET_SIZE,
        batch_size=Config.BATCH_SIZE
    )

    return train_generator, validation_generator

def build_nn_dataset_generator_InceptionV3():

    from keras.applications.InceptionV3 import preprocess_input
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
        # f'{Config.DATASET_DIR}/{Config.ENTITY_NAME}/train',
        os.path.join(Config.DATASET_DIR, Config.ENTITY_NAME, 'train'),
        target_size=Config.TARGET_SIZE,
        batch_size=Config.BATCH_SIZE
    )

    validation_generator = val_datagen.flow_from_directory(
        # f'{Config.DATASET_DIR}/{Config.ENTITY_NAME}/val',
        os.path.join(Config.DATASET_DIR, Config.ENTITY_NAME, 'test'),
        target_size=Config.TARGET_SIZE,
        batch_size=Config.BATCH_SIZE
    )

    return train_generator, validation_generator


def build_knn_dataset():
    train_data = []
    labels = []

    data_extensions = ['jpg', 'png']

    for item in list(os.walk(os.path.join(Config.DATASET_DIR, Config.ENTITY_NAME)))[2:]:

        if (item[0].split(os.path.sep)[2] == 'val') or (item[0].split(os.path.sep)[2] == 'test'):
            break

        if len(item[2]) > 0:
            label = item[0].split(os.path.sep)[-1]
            for image_name in tqdm(item[2]):
                if image_name.split('.')[-1] in data_extensions:

                    image = cv2.imread(os.path.join(item[0], image_name))

                    image = cv2.resize(image, (300, 300),
                                       interpolation=cv2.INTER_AREA)

                    train_data.append(image)

                    labels.append(label)

    return train_data, labels


if __name__ == "__main__":
    Config.ENTITY_NAME = 'animal'
    build_knn_dataset()
