import sys
sys.path.append('..')

from utils.config import Config, load_trained_model
from models import nn_model, knn_model
from utils.train import nn_train



Config.ENTITY_NAME = "Animal"
Config.ITERATION = 0
Config.NB_CLASSES = 3

msg, history = nn_train()
