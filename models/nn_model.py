from keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions
from keras.layers import Dense, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Activation, GlobalAveragePooling2D
from keras.models import Model, Sequential, load_model
from keras_preprocessing import image
from keras.callbacks import ReduceLROnPlateau
from config import config


class NNModel(object):

    def __init__(self, nb_classes):
        self.base_model = VGG19(weights='imagenet', include_top=False)
        self.nb_classes = nb_classes

    def _model(self):
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
        model = Model(inputs=self.base_model.input, outputs=predictions)
        return model

    def _setup_to_transfer_learn(self):
        pass

    def build_model(self):
        pass
