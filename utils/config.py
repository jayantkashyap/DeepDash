class Config:
    NB_CLASSES = 3
    EPOCHS = 3
    BATCH_SIZE = 32
    DATASET_DIR = 'repo'
    TARGET_SIZE = (224, 224)

    # TRAINING HYPERPARAMETERS

    # COMPILE
    OPTIMIZER = 'adam'
    LOSS = 'categorical_crossentropy'
    METRICS = ['accuracy']

    # MODEL
    MODEL_NAME = ''
    ITERATION = 0
    ENTITY_NAME = ''

    MODEL = None
    LABELS_TO_CLASSES = None


def load_trained_model(path_to_model):

    from keras.models import load_model
    import keras.backend as K
    import os

    K.clear_session()

    if not os.path.exists(path_to_model):
        return

    Config.MODEL = load_model(path_to_model)


def load_trained_classes(path_to_classes):
    import pickle
    import os

    if not os.path.exists(path_to_classes):
        return
    Config.CLASSES = pickle.load(open(path_to_classes, 'rb'))
