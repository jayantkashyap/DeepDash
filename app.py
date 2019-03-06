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

    print(Config.LABELS_TO_CLASSES)
    Config.ENTITY_NAME = "animal"
    _, status = train()

    return status


@app.route('/predict', methods=['GET', 'POST'])
def prediction():
    Config.ENTITY_NAME = 'animal'
    Config.ITERATION = 0
    Config.MODEL_NAME = 'nn_model'

    image = Image.open('cats_00001.jpg')
    preds = predict(image, entity_name='nist',
                    model_name='nn_model', model_iteration=0)

    print(preds)
    return ""


if __name__ == "__main__":
    app.run()
