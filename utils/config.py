class Config:
    NB_CLASSES = 10
    EPOCHS = 30
    BATCH_SIZE = 10
    DATASET_DIR = ''
    TARGET_SIZE = ()
    LOSS = ""

    # TRAINING HYPERPARAMETERS

    # CALLBACK
    PATIENCE = 3
    MIN_LEARNING_RATE = 0
    MONITOR = 'val_loss'

    # TRAINING
    OPTIMIZER = 'adam'
    LOSS = 'categorical_crossentropy'

    # MODEL
    MODEL_NAME = 'model'
    ITERATIONS = 0
    ENTITY_NAME = ''
