class Config:
    NB_CLASSES = 3
    EPOCHS = 1
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
    DEFAULT_GRAPH = None

    MODEL = None
    LABELS_TO_CLASSES = None


def load_trained_model(path_to_model):

    from keras.models import load_model
    import keras.backend as K
    import tensorflow as tf
    import os

    K.clear_session()
    tf.reset_default_graph()

    if not os.path.exists(path_to_model):
        return

    Config.MODEL = load_model(path_to_model)
    Config.DEFAULT_GRAPH = tf.get_default_graph()


def load_trained_classes(path_to_classes):
    import pickle
    import os

    if not os.path.exists(path_to_classes):
        return
    Config.CLASSES = pickle.load(open(path_to_classes, 'rb'))
