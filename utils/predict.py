from keras.preprocessing.image import img_to_array
from keras.applications.vgg19 import preprocess_input
import tensorflow as tf
import numpy as np
import pickle
import sys
import os

sys.path.append('..')
from utils.config import Config, load_trained_model


def preprocess_image(image):
    if image.mode != "RGB":
        image.convert("RGB")

    image = image.resize(Config.TARGET_SIZE)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    return image


def predict(image, entity_name, model_name, model_iteration):
    message = ""

    if (Config.MODEL == None):
        if os.path.exists(f"data/{entity_name}/{model_name}_{model_iteration}.model"):
            load_trained_model(
                f"data/{entity_name}/{model_name}_{model_iteration}")
        else:
            return None, None, 1, "No Model Trained"

    if entity_name != Config.ENTITY_NAME or\
            model_name != Config.MODEL or \
            model_iteration != Config.ITERATION:

        load_trained_model(
            f"data/{entity_name}/{model_name}_{model_iteration}")

    image = preprocess_image(image)

    try:
        with Config.DEFAULT_GRAPH.as_default():
            preds = Config.MODEL.predict(image)
    except Exception:
        status = 2
        return "", None,

    return Config.LABELS_TO_CLASSES[np.argmax(preds)],\
        round(np.argmax(preds)*100, 2), status, message
