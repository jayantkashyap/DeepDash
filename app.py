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
    # print(Config.MODEL.summary())
    print()
    print(Config.MODEL_NAME)
    return status


if __name__ == "__main__":
    load_trained_model('')
    app.run()
