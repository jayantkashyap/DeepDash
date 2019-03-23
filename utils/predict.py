import numpy as np
import pickle
import sys
import os

sys.path.append('..')
from utils.config import Config, load_trained_model


def preprocess_image(image):
    from keras.preprocessing.image import img_to_array
    from keras.applications.vgg19 import preprocess_input

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
            # print(os.path.exists(
            #     f"data/{entity_name}/{model_name}_{model_iteration}.model"))
            load_trained_model(
                f"data/{entity_name}/{model_name}_{1}")
        else:
            return "No Model Trained"

    if entity_name != Config.ENTITY_NAME or\
            model_name != Config.MODEL or \
            model_iteration != Config.ITERATION:

        load_trained_model(
            f"data/{entity_name}/{model_name}_{model_iteration}")

    image = preprocess_image(image)

    with Config.DEFAULT_GRAPH.as_default():
        preds = Config.MODEL.predict(image)

    return f"{Config.LABELS_TO_CLASSES[np.argmax(preds)]} with Probability {np.max(preds)*100}"
