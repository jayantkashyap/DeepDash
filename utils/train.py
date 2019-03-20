from keras.preprocessing import image
from keras.callbacks import ReduceLROnPlateau, BaseLogger, ProgbarLogger, RemoteMonitor
from keras.applications.vgg19 import preprocess_input
import tensorflow as tf
from PIL import Image
import numpy as np
import pickle
import sys
import cv2
import os

sys.path.append('..')
from utils.config import Config
from models.nn_model import NNModel
from utils.data_generator import build_nn_dataset_generator


def train():

    import keras.backend as K
    K.clear_session()
    tf.reset_default_graph()

    model = NNModel().build()
    reduce_lr_callback = ReduceLROnPlateau(monitor='val_loss',
                                           factor=0.1,
                                           patience=3,
                                           min_lr=0)

    # train_generator, validatation_generator = build_nn_dataset_generator()
    train_generator = build_nn_dataset_generator()

    model.compile(optimizer=Config.OPTIMIZER,
                  loss=Config.LOSS, metrics=Config.METRICS)

    history = model.fit_generator(train_generator,
                                  steps_per_epoch=train_generator.n//train_generator.batch_size,
                                  epochs=Config.EPOCHS,
                                  #   validation_data=validatation_generator,
                                  #   validation_steps=validatation_generator.n//validatation_generator.batch_size,
                                  class_weight='auto',
                                  callbacks=[reduce_lr_callback])

    Config.MODEL = model
    Config.DEFAULT_GRAPH = tf.get_default_graph()

    Config.MODEL_NAME = "nn_model"
    Config.LABELS_TO_CLASSES = {v: k for k,
                                v in train_generator.class_indices.items()}

    print(Config.LABELS_TO_CLASSES)

    if not os.path.isdir(f'data/{Config.ENTITY_NAME}'):
        print(os.path.isdir(f'data/{Config.ENTITY_NAME}'))
        os.makedirs(f'data/{Config.ENTITY_NAME}')

    model.save(
        f'data/{Config.ENTITY_NAME}/{Config.MODEL_NAME}_{Config.ITERATION}.model')

    pickle.dump(Config.LABELS_TO_CLASSES, open(
        f'data/{Config.ENTITY_NAME}/{Config.MODEL_NAME}_{Config.ITERATION}_classes.p', 'wb'))

    return history, "Model Trained Successfully!"
