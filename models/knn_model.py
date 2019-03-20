from collections import Counter
import numpy as np
import pickle
import sys
import os


sys.path.append('..')
from utils.config import Config
from utils.data_generator import build_knn_dataset


class KNN_Model(object):

    def __init__(self, k, distance):
        self.model = None
        self.k = k
        self.distance = 'L2'

    def train(self):
        from sklearn.preprocessing import LabelEncoder

        train_data, labels = build_knn_dataset()

        self.x_train = np.array(list(map(np.ravel, np.array(train_data))))

        le = LabelEncoder()
        self.y_train = le.fit_transform(labels)

        Config.LABELS_TO_CLASSES = {i: c for i, c in enumerate(le.classes_)}
        Config.MODEL == self.x_train, self.y_train
        Config.MODEL_NAME = 'knn_model'

        if not os.path.isdir(f'data/{Config.ENTITY_NAME}'):
            print(os.path.isdir(f'data/{Config.ENTITY_NAME}'))
            os.makedirs(f'data/{Config.ENTITY_NAME}')

        pickle.dump((self.x_train, self.y_train), open(
            f'data/{Config.ENTITY_NAME}/{Config.MODEL_NAME}_{Config.ITERATION}.p', 'wb'))

        pickle.dump(Config.LABELS_TO_CLASSES, open(
            f'data/{Config.ENTITY_NAME}/{Config.MODEL_NAME}_{Config.ITERATION}_classes.p', 'wb'))

    def l1_distance(self, image, image_array):
        return np.sum(np.abs(image_array - image), axis=1)

    def l2_distance(self, image, image_array):
        return np.sqrt(np.sum(np.square(image_array - image), axis=1))

    def predict(self, image, entity_name, model_name, model_iteration):

        if Config.MODEL == None:
            if os.path.exists(f"data/{entity_name}/{model_name}_{model_iteration}.p"):
                Config.MODEL = pickle.load(
                    open(f"data/{entity_name}/{model_name}_{model_iteration}.p", 'rb'))
                Config.LABELS_TO_CLASSES = pickle.load(
                    open(f"data/{entity_name}/{model_name}_{model_iteration}_classes.p", 'rb'))
                Config.MODEL_NAME = "knn_model"
            else:
                return None, None, 1, "No Model Trained"

        if entity_name != Config.ENTITY_NAME or\
                model_name != Config.MODEL or \
                model_iteration != Config.ITERATION:

            Config.MODEL = pickle.load(
                open(f"data/{entity_name}/{model_name}_{model_iteration}.p", 'rb'))
            Config.LABELS_TO_CLASSES = pickle.load(
                open(f"data/{entity_name}/{model_name}_{model_iteration}_classes.p", 'rb'))
            Config.MODEL_NAME = "knn_model"

        self.x_train, self.y_train = Config.MODEL

        if self.distance == 'L1':
            k_preds = np.argsort(self.l1_distance(
                np.ravel(image), self.x_train))[:self.k]            

        if self.distance == 'L2':
            k_preds = np.argsort(self.l2_distance(
                np.ravel(image), self.y_train))[:self.k]
        
        lables = [self.y_train[i] for i in k_preds]
        labels = [Config.LABELS_TO_CLASSES.get(i) for i in labels]
        prediction = Counter(labels).most_common(1)[0][0]

        return prediction

    


