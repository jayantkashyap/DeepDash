from keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions
from keras.layers import Dense, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Activation, GlobalAveragePooling2D
from keras.models import Model, Sequential, load_model
from keras.preprocessing import image
from keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
import pandas as pd

import cv2
import os


class ClassifierEngine(object):
    """docstring for ClassifierEngine"""

    def __init__(self):
        self.model = None
        self.base_model = None
        self.classes = None
        self.loss = None
        self.optimizer = None
        self.dataset_dir = None
        self.batch_size = None
        self.target_size = None
        self.train_generator = None
        self.test_generator = None
        self.label_to_class = {}

    def _build_dataset_generator(self):

        _train_datagen = image.ImageDataGenerator(
            preprocessing_function=preprocess_input,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2)

        _val_datagen = image.ImageDataGenerator(
            preprocessing_function=preprocess_input,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2)

        self.train_generator = _train_datagen.flow_from_directory(
            f'{self.dataset_dir}/train',
            self.target_size,
            batch_size=self.batch_size,)

        self.validation_generator = _val_datagen.flow_from_directory(
            f'{self.dataset_dir}/val',
            self.target_size,
            batch_size=self.batch_size,)

        for k, v in self.train_generator.class_indices.items():
            self.label_to_class[v] = k

    def build(self, dataset_dir='dataset',
              batch_size=16,
              target_size=(256, 256),
              optimizer='adam',
              loss='categorical_crossentropy'):

        self.target_size = target_size
        self.batch_size = batch_size
        self.dataset_dir = dataset_dir
        self.optimizer = optimizer
        self.loss = loss

        self._build_dataset_generator()
        self.classes = self.train_generator.classes
        self.base_model = VGG19(weights='imagenet', include_top=False)
        self.model = self._add_new_last_layer()
        self._setup_to_transfer_learn(self.optimizer, self.loss)

    def _add_new_last_layer(self):
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
        predictions = Dense(
            len(self.train_generator.class_indices.keys()), activation='softmax')(x)
        model = Model(inputs=self.base_model.input, outputs=predictions)
        return model

    def _setup_to_transfer_learn(self, optimizer, loss):
        for layer in self.base_model.layers:
            layer.trainable = False

        self.model.compile(optimizer=optimizer,
                           loss=loss,
                           metrics=['accuracy'])

    def train(self, epochs, save_model=True, plot=False):
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                      patience=3, min_lr=0)

        history = self.model.fit_generator(
            self.train_generator,
            steps_per_epoch=self.train_generator.n//self.train_generator.batch_size,
            epochs=epochs,
            validation_data=self.validation_generator,
            validation_steps=self.validation_generator.n//self.validation_generator.batch_size,
            class_weight='auto',
            callbacks=[reduce_lr])
        if plot:
            self._plot_training(history)

        if save_model:
            self.model.save('model.model')

    def _plot_training(slef, history):
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(len(acc))

        plt.plot(epochs, acc, 'r')
        plt.plot(epochs, val_acc, 'b')
        plt.title('Training and validation accuracy')
        plt.legend(['Accuracy', 'Validation Accuracy'], loc='lower right')
        plt.savefig('Accuracy Plot')
        plt.figure()
        plt.plot(epochs, loss, 'r')
        plt.plot(epochs, val_loss, 'b')
        plt.title('Training and validation loss')
        plt.legend(['Loss', 'Val Loss'], loc='upper right')
        plt.savefig('Loss Plot')
        plt.show()

    def predict(self, model, img, target_size):
        if img.size != target_size:
            img = img.resize(target_size)

        x = np.array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        preds = model.predict(x)
        return preds

    def evaluate(self, confusion_matrix, save_plot=True):
        df_cm = pd.DataFrame(conf_matrix, index=list(self.label_to_class.values()),
                             columns=list(self.label_to_class.values()))
        plt.figure(figsize=(10, 7))
        plot = sn.heatmap(df_cm, annot=True)
        plt.show()
        plot.figure.savefig('ConfusionMatrix.png')

    def load_pretrained_model(self, model):
        return load_model(model)

    def __getitem__(self):
        return self.model

    def __repr__(self):
        return "VGG19 Model Transfer Learning"


if __name__ == '__main__':
    # ENGINE INTIALIZATION
    classifier_engine = ClassifierEngine()

    # ENGINE TRAINING
    classifier_engine.build()
    classifier_engine.train(epochs=3, plot=True)

    # ENGINE TESTING
    # classifier_engine._build_dataset_generator()
    # model = classifier_engine.load_pretrained_model('models/model_11.model')
    # fnames = os.listdir('dataset/test')

    # for category in fnames:
    #     for image in os.listdir(os.path.join('dataset', 'test', category)):
    #         img = Image.open(os.path.join('dataset', 'test', category, image))
    #         p = classifier_engine.predict(model, img, (256,256))
    #         plt.imshow(img)
    #         plt.title(f'{classifier_engine.label_to_class[np.argmax(p)]} | {max(p[0])*100:.2f}')
    #         plt.show()

    # fnames = os.listdir('backbar_testing_images')
    # for image in fnames:
    #     img = Image.open(os.path.join('backbar_testing_images', image))
    #     p = classifier_engine.predict(model, img, (256,256))
    #     plt.imshow(img)
    #     plt.title(f'{classifier_engine.label_to_class[np.argmax(p)]} | {max(p[0])*100:.2f}')
    #     plt.show()

    # EVALUATION

    # y_true = []
    # y_pred = []

    # fnames = os.listdir(os.path.join('dataset', 'test'))
    # for category in fnames:
    #     for file in os.listdir(os.path.join('dataset', 'test', category)):
    #             y_true.append(category)

    # for category in fnames:
    #     for file in os.listdir(os.path.join('dataset', 'test', category)):
    #             img = Image.open(os.path.join('dataset', 'test', category, file))
    #             p = classifier_engine.predict(model, img, (256,256))
    #             y_pred.append(classifier_engine.label_to_class[np.argmax(p)])

    # conf_matrix = confusion_matrix(y_true, y_pred)
    # classifier_engine.evaluate(conf_matrix)
