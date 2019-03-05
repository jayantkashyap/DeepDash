from flask import Flask, request, jsonify
from models.handle import handle
from utils.config import Config, load_trained_classes, load_trained_model
from utils.train import train
from utils.predict import predict
from PIL import Image
import numpy as np
import sys
import io
import os


app = Flask(__name__)


@app.route('/train', methods=['GET', 'POST'])
def training():
    import keras.backend as K
    K.clear_session()

    print(Config.LABELS_TO_CLASSES)
    Config.ENTITY_NAME = "animal"
    _, status = train()
    print(Config.MODEL.summary())
    print(Config.MODEL_NAME)
    print(Config.LABELS_TO_CLASSES)
    return status


@app.route('/predict', methods=['GET', 'POST'])
def prediction():
    Config.ENTITY_NAME = 'nist'
    Config.ITERATION = 0
    Config.MODEL_NMAE = 'nn_model'
    image = Image.open('cats_00001.jpg')
    predict(image, entity_name='temp', model_name='temp', model_iteration=0)


if __name__ == "__main__":
    import keras.backend as K

    K.clear_session()

    load_trained_model('')
    app.run()
