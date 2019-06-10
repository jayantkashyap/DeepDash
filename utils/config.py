class Config:
    NB_CLASSES = 3
    EPOCHS = 10
    BATCH_SIZE = 32
    DATASET_DIR = 'repo'
    MODEL_DIR = '/data'
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


def load_trained_model(path):

    from keras.models import load_model
    import keras.backend as K
    import tensorflow as tf
    import pickle
    import os

    K.clear_session()
    tf.reset_default_graph()

    if not os.path.exists(f"{path}.model"):
        return

    Config.MODEL = load_model(f"{path}.model")
    Config.DEFAULT_GRAPH = tf.get_default_graph()
    Config.LABELS_TO_CLASSES = pickle.load(open(f"{path}_classes.p", 'rb'))


# def load_trained_classes(path_to_classes):
#     import pickle
#     import os

#     if not os.path.exists(path_to_classes):
#         return
#     Config.CLASSES = pickle.load(open(path_to_classes, 'rb'))
