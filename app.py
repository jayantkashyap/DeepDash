from flask import Flask, request, jsonify, render_template
from werkzeug import secure_filename
from models.handle import handle
from utils.config import Config, load_trained_model
from utils.train import train
from utils.predict import predict
from PIL import Image
import numpy as np
import sys
import io
import os


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/train', methods=['GET', 'POST'])
def training():

    Config.ENTITY_NAME = "animal"
    _, status = train()

    return status


@app.route('/predict', methods=['GET', 'POST'])
def prediction():
    result = {}

    Config.ENTITY_NAME = 'animal'
    Config.ITERATION = 0
    Config.MODEL_NAME = 'nn_model'

    image = Image.open('cats_00001.jpg')
    preds = predict(image, entity_name='nist',
                    model_name='nn_model', model_iteration=0)

    if preds is not None:
        result["Class"] = preds[0]
        result["Probability"] = preds[1]
        # result["Image_Name"] = Image_Name
        result["Message"] = "Successfull Prediction"
        result["Status"] = 0
    else:
        result["Class"] = ""
        result["Probability"] = None
        # result["Image_Name"] = Image_Name
        result["Message"] = "Successfull Prediction"
        result["Status"] = 0

    print(preds)
    return jsonify(result)


@app.route('/upload', methods=['POST'])
def upload():
    data = request.form['entity_name']
    print(data)
    files = request.files.getlist('files[]')
    if not os.path.exists(data):
        os.makedirs(data)

    for f in files:
        f.save(os.path.join(data, secure_filename(f.filename)))
    return "Files Uploaded Successfully!!"


if __name__ == "__main__":
    app.run(host="0.0.0.0")
