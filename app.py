from flask import Flask, request, jsonify, render_template
from werkzeug import secure_filename
from models import knn_model, nn_model
from utils.config import Config, load_trained_model
from utils.train import nn_train_VGG19, nn_train_InceptionV3
from utils.predict import predict
from PIL import Image
import base64
import numpy as np
import sys
import json
import io
import os

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/train', methods=['GET', 'POST'])
def training():

    if request.get_json()['modelName'] == 'KNN':
        Config.ENTITY_NAME = request.get_json()['entityName']
        Config.ITERATION = int(request.get_json()['iteration'])
        Config.NB_CLASSES = len(os.listdir(os.path.join(
            Config.DATASET_DIR, Config.ENTITY_NAME, 'train')))

        msg, history = nn_train_InceptionV3()

        return jsonify({"Message": msg, "History": history})

    if request.get_json()['modelName'] == 'DNN':
        Config.ENTITY_NAME = request.get_json()['entityName']
        Config.ITERATION = int(request.get_json()['iteration'])
        Config.NB_CLASSES = len(os.listdir(os.path.join(
            Config.DATASET_DIR, Config.ENTITY_NAME, 'train')))

        msg, history = nn_train_VGG19()

        return jsonify({"Message": msg, "History": history})
        # return msg


@app.route('/predict', methods=['GET', 'POST'])
def prediction():

    Config.ENTITY_NAME = request.get_json()['entityName']
    Config.ITERATION = request.get_json()['iteration']
    Config.MODEL_NAME = request.get_json()['modelName']
    image = request.get_json()['image']

    image = base64.b64decode(str(image))
    image = Image.open(io.BytesIO(image))

    # if model_name == 'KNN':
    #     preds = knn_model.KNN_Model(), predict(
    #         image, entity_name, 'KNN', model_iteration)
    # elif model_name == 'DNN':

    if Config.MODEL_NAME == 'DNN':
        preds = predict(image, entity_name=Config.ENTITY_NAME,
                        model_name=Config.MODEL_NAME, model_iteration=Config.ITERATION)
    if Config.MODEL_NAME == 'KNN':
        preds = "Model not Trained yet"
    # print(preds)
    return jsonify(preds)


@app.route('/upload', methods=['POST'])
def upload():
    data = request.form['entity_name']
    print(data)
    files = request.files.getlist('files[]')
    if not os.path.exists(data):
        os.makedirs(data)

    for f in files:
        print(f)
        f.save(os.path.join(data, secure_filename(f.filename)))
    return "Files Uploaded Successfully!!"


if __name__ == "__main__":
    # import keras
    # import tensorflow as tf

    app.run(host="0.0.0.0", port=4500)
