from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
import numpy as np
import pickle
import sys

sys.path.append('..')
from utils.config import Config, load_trained_model, load_trained_classes


def preprocess_image(image):
    if image.mode != "RGB":
        image.convert("RGB")

    image = image.resize(Config.TARGET_SIZE)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    return image


def predict(image, entity_name, model_name, model_iteration):
    if model_name != Config.MODEL or \
            model_iteration != Config.ITERATION:
        Config.MODEL = load_trained_model("")
        Config.LABELS_TO_CLASSES = load_trained_classes("")

    image = preprocess_image(image)
    preds = Config.MODEL.predict(image)

    return preds
