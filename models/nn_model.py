"""Convolution Neural Network Model"""

import sys
sys.path.append('..')
from utils.config import Config

class NNModel_VGG19(object):

    def __init__(self):

        from keras.applications.vgg19 import VGG19

        self.base_model = VGG19(weights='imagenet', include_top=False)
        self.nb_classes = Config.NB_CLASSES
        self.model = None

    def _model(self):

        from keras.layers import Dense, BatchNormalization, Dropout, \
            Activation, GlobalAveragePooling2D
        from keras.models import Model

        x = self.base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(1024)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(self.nb_classes, activation='softmax')(x)
        self.model = Model(inputs=self.base_model.input, outputs=predictions)

    def _setup_to_transfer_learn(self):
        for layer in self.base_model.layers:
            layer.trainable = False

    def build(self):
        self._setup_to_transfer_learn()
        self._model()
        return self.model


class NNModel_InceptionV3(object):

    def __init__(self):

        from keras.applications.inception_v3 import InceptionV3

        self.base_model = InceptionV3(weights='imagenet', include_top=False)
        self.nb_classes = Config.NB_CLASSES
        self.model = None

    def _model(self):

        from keras.layers import Dense, BatchNormalization, Dropout, \
            Activation, GlobalAveragePooling2D
        from keras.models import Model

        x = self.base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(1024)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(self.nb_classes, activation='softmax')(x)
        self.model = Model(inputs=self.base_model.input, outputs=predictions)

    def _setup_to_transfer_learn(self):
        for layer in self.base_model.layers:
            layer.trainable = False

    def build(self):
        self._setup_to_transfer_learn()
        self._model()
        return self.model


# if __name__ == "__main__":
#     from keras import backend as K

#     model = NNModel().build()
#     print(model.summary())

#     K.clear_session()
