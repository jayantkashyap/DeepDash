from keras.models import load_model
import pickle


class Config:
    NB_CLASSES = 10
    EPOCHS = 30
    BATCH_SIZE = 10
    DATASET_DIR = ''
    TARGET_SIZE = ()
    LOSS = ""

    # TRAINING HYPERPARAMETERS

    # TRAINING
    OPTIMIZER = 'adam'
    LOSS = 'categorical_crossentropy'

    # MODEL
    MODEL_NAME = 'model'
    ITERATION = 0
    ENTITY_NAME = ''

    MODEL = None
    LABELS_TO_CLASSES = None


def load_trained_model(path_to_model):
    Config.MODEL = load_model(path_to_model)


def load_trained_classes(path_to_classes):
    Config.CLASSES = pickle.load(open(path_to_classes, 'rb'))
