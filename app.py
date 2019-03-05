from flask import Flask, request, jsonify
from models.handle import handle
from utils.config import Config, load_trained_classes, load_trained_model
from utils.train import train
from PIL import Image
import numpy as np
import sys
import io
import os


app = Flask(__name__)


@app.route('/train', methods=['GET', 'POST'])
def training():
    Config.ENTITY_NAME = "animals"
    _, status = train()
    print(Config.MODEL.summary())
    print(Config.MODEL_NAME)
    print(Config.LABELS_TO_CLASSES)
    return status


if __name__ == "__main__":
    import keras.backend as K

    K.clear_session()

    load_trained_model('')
    app.run()
