"""KNN Classifier Model"""

from collections import Counter
import numpy as np
import pickle
import sys
import os


sys.path.append('..')
from utils.data_generator import build_knn_dataset
from utils.config import Config

class KNN_Model(object):

    def __init__(self, k, distance='L2'):
        """
        KNN Classifier

        Parameters:
        __________

        k        (int): Specify the number of nearest neighbor to consider.
        distance (str): Specify the distance metric. Either Euclidean distance or Absolute distance

        """
        self.model = None
        self.k = k
        self.distance = distance
        self.x_train, self.y_train = None, None

    def train(self):
        """
        Training for the KNN model. Creates the model by memorizing all the training dataset, 
        i.e., model is all the training data and its labels.

        """
        from sklearn.preprocessing import LabelEncoder

        # build dataset
        train_data, labels = build_knn_dataset()

        self.x_train = np.array(list(map(np.ravel, np.array(train_data))))

        # Labels needs to be encoded to integers
        le = LabelEncoder()
        self.y_train = le.fit_transform(labels)

        Config.LABELS_TO_CLASSES = {i: c for i, c in enumerate(le.classes_)}
        Config.MODEL = (self.x_train, self.y_train)
        Config.MODEL_NAME = 'KNN'

        # To save the model check if a folder exists with given folder name
        if not os.path.isdir(f'data/{Config.ENTITY_NAME}'):
            print(os.path.isdir(f'data/{Config.ENTITY_NAME}'))
            os.makedirs(f'data/{Config.ENTITY_NAME}')

        # Store model
        pickle.dump((self.x_train, self.y_train), open(
            f'data/{Config.ENTITY_NAME}/{Config.MODEL_NAME}_{Config.ITERATION}.p', 'wb'))

        # Store classes
        pickle.dump(Config.LABELS_TO_CLASSES, open(
            f'data/{Config.ENTITY_NAME}/{Config.MODEL_NAME}_{Config.ITERATION}_classes.p', 'wb'))

        return "Model Trained Successfully!"

    def predict(self, image, entity_name, model_name, model_iteration):
        """
        Prediction for KNN Classifier. The model makes prediction by checking the distance
        of image to all the images in the training set and returns the label to the image
        to which the distance is least.

        Parameters:
        __________

        image               (numpy.ndarray) : Image to make prediction on.
        entity_name         (str)           : Specifies the entity on which prediction has to be made.
        model_name          (str)           : Specifies model name from already trained model to make predictions on.
        model_iteration     (int)           : Specifies iteration number to load the model with specified iteration number.

        Returns:
        ________

        prediction          (str): The class to which the image is most closely related to

        """

        # Load model
        if Config.MODEL is None:
            if os.path.exists(f"data/{entity_name}/{model_name}_{model_iteration}.p"):
                Config.MODEL = pickle.load(
                    open(f"data/{entity_name}/{model_name}_{model_iteration}.p", 'rb'))
                Config.LABELS_TO_CLASSES = pickle.load(
                    open(f"data/{entity_name}/{model_name}_{model_iteration}_classes.p", 'rb'))
                Config.MODEL_NAME = "KNN"
            else:
                return None, None, 1, "No Model Trained"

        if entity_name != Config.ENTITY_NAME or\
                model_name != Config.MODEL or \
                model_iteration != Config.ITERATION:

            Config.MODEL = pickle.load(
                open(f"data/{entity_name}/{model_name}_{model_iteration}.p", 'rb'))
            Config.LABELS_TO_CLASSES = pickle.load(
                open(f"data/{entity_name}/{model_name}_{model_iteration}_classes.p", 'rb'))
            Config.MODEL_NAME = "KNN"

        self.x_train, self.y_train = Config.MODEL

        if self.distance == 'L1':
            k_preds = np.argsort(l1_distance(
                np.ravel(image), self.x_train))[:self.k]

        if self.distance == 'L2':
            k_preds = np.argsort(l2_distance(
                np.ravel(image), self.y_train))[:self.k]

        lables = [self.y_train[i] for i in k_preds]
        lables = [Config.LABELS_TO_CLASSES.get(i) for i in lables]
        prediction = Counter(lables).most_common(1)[0][0]

        return prediction


def l1_distance(image, image_array):
    """
    Returns the distances of an image from all the images in the image_array.

    Calculates absolute distance, i.e., d = |x_2 - x_1|

    Parameters:
    __________

    image               (numpy.ndarray) : Query Image.
    image_array         (numpy.ndarray) : Dataset of images

    Returns:
    ________

    numpy.array : Distance of the image from all the images.
    """
    return np.sum(np.abs(image_array - image), axis=1)


def l2_distance(image, image_array):
    """
    Returns the euclidean distances of an image from all the images in the image_array.

    Calculates absolute distance, i.e., d = sqrt(|x_2 - x_1|*|x_2 - x_1|)

    Parameters:
    __________

    image               (numpy.ndarray) : Query Image.
    image_array         (numpy.ndarray) : Dataset of images

    Returns:
    ________

    numpy.array : Euclidean distance of the image from all the images.
    """
    return np.sqrt(np.sum(np.square(image_array - image), axis=1))
