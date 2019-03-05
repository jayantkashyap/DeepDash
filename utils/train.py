from keras.preprocessing import image
from keras.callbacks import ReduceLROnPlateau, BaseLogger, ProgbarLogger, RemoteMonitor
from keras.applications.vgg19 import preprocess_input
from PIL import Image
import pickle
import sys
import cv2
import os

sys.path.append('..')
from utils.config import Config
from models.nn_model import NNModel


def build_dataset_generator():

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


def train():
    model = NNModel().build()
    reduce_lr_callback = ReduceLROnPlateau(monitor='val_loss',
                                           factor=0.1,
                                           patience=3,
                                           min_lr=0)

    train_generator, validatation_generator = build_dataset_generator()

    model.compile(optimizer=Config.OPTIMIZER,
                  loss=Config.LOSS, metrics=Config.METRICS)

    history = model.fit_generator(train_generator,
                                  steps_per_epoch=train_generator.n//train_generator.batch_size,
                                  epochs=Config.EPOCHS,
                                  validation_data=validatation_generator,
                                  validation_steps=validatation_generator.n//validatation_generator.batch_size,
                                  class_weight='auto',
                                  callbacks=[reduce_lr_callback])

    Config.MODEL = model
    Config.MODEL_NAME = "nn_model"

    if not os.path.isdir(f'../data/{Config.ENTITY_NAME}'):
        os.makedirs(f'../data/{Config.ENTITY_NAME}')

    model.save(
        f'../data/{Config.ENTITY_NAME}/{Config.MODEL_NAME}_{Config.ITERATION}.h5')

    pickle.dump(train_generator.classes, open(
        f'../data/{Config.ENTITY_NAME}/{Config.MODEL_NAME}_{Config.ITERATION}_classes.p', 'wb'))
    # history = None
    return history, "Model Trained Successfully!"
