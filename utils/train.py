from PIL import Image
import numpy as np
import pickle
import sys
import cv2
import os

sys.path.append('..')
from utils.data_generator import build_nn_dataset_generator_VGG19, build_nn_dataset_generator_InceptionV3
from models.nn_model import NNModel_VGG19, NNModel_InceptionV3
from utils.config import Config


def nn_train_VGG19():
    from keras.callbacks import ReduceLROnPlateau
    import tensorflow as tf

    import keras.backend as K
    K.clear_session()
    tf.reset_default_graph()

    model = NNModel_VGG19().build()
    reduce_lr_callback = ReduceLROnPlateau(monitor='val_loss',
                                           factor=0.1,
                                           patience=3,
                                           min_lr=0)

    # train_generator, validatation_generator = build_nn_dataset_generator()
    train_generator, validation_generator = build_nn_dataset_generator_VGG19()

    model.compile(optimizer=Config.OPTIMIZER,
                  loss=Config.LOSS, metrics=Config.METRICS)

    history = model.fit_generator(train_generator,
                                  steps_per_epoch=train_generator.n//train_generator.batch_size,
                                  epochs=Config.EPOCHS,
                                  validation_data=validation_generator,
                                  validation_steps=validation_generator.n//validation_generator.batch_size,
                                  class_weight='auto',
                                  callbacks=[reduce_lr_callback])

    Config.MODEL = model
    Config.DEFAULT_GRAPH = tf.get_default_graph()

    Config.MODEL_NAME = "DNN"
    Config.LABELS_TO_CLASSES = {v: k for k,
                                v in train_generator.class_indices.items()}

    # print(Config.LABELS_TO_CLASSES)

    if not os.path.isdir(os.path.join(Config.MODEL_DIR, Config.ENTITY_NAME)):
        print(os.path.isdir(os.path.join(Config.MODEL_DIR, Config.ENTITY_NAME)))
        os.makedirs(os.path.join(Config.MODEL_DIR, Config.ENTITY_NAME))

    model.save(
        f'{Config.MODEL_DIR}/{Config.ENTITY_NAME}/{Config.MODEL_NAME}_{Config.ITERATION}.model')

    pickle.dump(Config.LABELS_TO_CLASSES, open(
        f'{Config.MODEL_DIR}/{Config.ENTITY_NAME}/{Config.MODEL_NAME}_{Config.ITERATION}_classes.p', 'wb'))

    val_loss = history.history['val_loss']
    val_loss = list(map(lambda x: x.item(), val_loss))

    val_acc = history.history['val_acc']
    val_acc = list(map(lambda x: x.item(), val_acc))

    loss = history.history['loss']
    loss = list(map(lambda x: x.item(), loss))

    acc = history.history['acc']
    acc = list(map(lambda x: x.item(), acc))

    print(val_acc)
    return "Model Trained Successfully!", {"val_loss": val_loss, "val_acc": val_acc, "loss": loss, "acc": acc}


def nn_train_InceptionV3():
    from keras.callbacks import ReduceLROnPlateau
    import tensorflow as tf

    import keras.backend as K
    K.clear_session()
    tf.reset_default_graph()

    model = NNModel_InceptionV3().build()
    reduce_lr_callback = ReduceLROnPlateau(monitor='val_loss',
                                           factor=0.1,
                                           patience=3,
                                           min_lr=0)

    # train_generator, validatation_generator = build_nn_dataset_generator()
    train_generator, validation_generator = build_nn_dataset_generator_InceptionV3()

    model.compile(optimizer=Config.OPTIMIZER,
                  loss=Config.LOSS, metrics=Config.METRICS)

    history = model.fit_generator(train_generator,
                                  steps_per_epoch=train_generator.n//train_generator.batch_size,
                                  epochs=Config.EPOCHS,
                                  validation_data=validation_generator,
                                  validation_steps=validation_generator.n//validation_generator.batch_size,
                                  class_weight='auto',
                                  callbacks=[reduce_lr_callback])

    Config.MODEL = model
    Config.DEFAULT_GRAPH = tf.get_default_graph()

    Config.MODEL_NAME = "DNN"
    Config.LABELS_TO_CLASSES = {v: k for k,
                                v in train_generator.class_indices.items()}

    # print(Config.LABELS_TO_CLASSES)

    if not os.path.isdir(os.path.join(Config.MODEL_DIR, Config.ENTITY_NAME)):
        print(os.path.isdir(os.path.join(Config.MODEL_DIR, Config.ENTITY_NAME)))
        os.makedirs(os.path.join(Config.MODEL_DIR, Config.ENTITY_NAME))

    model.save(
        f'{Config.MODEL_DIR}/{Config.ENTITY_NAME}/{Config.MODEL_NAME}_{Config.ITERATION}.model')

    pickle.dump(Config.LABELS_TO_CLASSES, open(
        f'{Config.MODEL_DIR}/{Config.ENTITY_NAME}/{Config.MODEL_NAME}_{Config.ITERATION}_classes.p', 'wb'))

    val_loss = history.history['val_loss']
    val_loss = list(map(lambda x: x.item(), val_loss))

    val_acc = history.history['val_acc']
    val_acc = list(map(lambda x: x.item(), val_acc))

    loss = history.history['loss']
    loss = list(map(lambda x: x.item(), loss))

    acc = history.history['acc']
    acc = list(map(lambda x: x.item(), acc))

    print(val_acc)
    return "Model Trained Successfully!", {"val_loss": val_loss, "val_acc": val_acc, "loss": loss, "acc": acc}
