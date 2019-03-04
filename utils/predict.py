from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
import numpy as np
import sys

sys.path.append('..')
from utils.config import Config


def preprocess_image(image):
    if image.mode != "RGB":
        image.convert("RGB")

    image = image.resize(Config.TARGET_SIZE)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    return image


def predict(image, iteration):
    pass
