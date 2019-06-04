import sys
sys.path.append('..')

from utils.train import nn_train
from models import nn_model, knn_model
from utils.config import Config, load_trained_model


Config.ENTITY_NAME = "animal"
Config.ITERATION = 0
Config.NB_CLASSES = 3

msg, history = nn_train()
print(msg)
print(history.history)