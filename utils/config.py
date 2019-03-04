from keras.models import load_model


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


def load_model():

    # Config.MODEL = load_model
    pass
