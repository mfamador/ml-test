import logging
import os
import pickle

import keras.models as keras
import numpy as np
import tensorflow as tf

MODEL_NAME = "model.h5"
SCALER_NAME = "scaler.p"
TOKENIZER_NAME = "tokenizer.p"
ENCODER_NAME = "encoder.p"


def load_model():
    resource_path = os.environ.get('RESOURCE_PATH', 'resources')
    model = Model(resource_path)
    logging.info("Model loaded: {0}".format(model))
    return model


class Model:

    def __init__(self, resource_path):
        self.resource_path = resource_path
        self.model = self.__load_model()
        self.scaler = self.__load_scaler()
        self.tokenizer = self.__load_tokenizer()
        self.encoder = self.__load_encoder()
        self.graph = tf.get_default_graph()

    def predict(self, feature):
        feature_vector = self.__transform_input(feature)
        with self.graph.as_default():
            model_prediction = self.model.predict(feature_vector)
        return self.__transform_output(model_prediction)

    def __transform_input(self, feature):

        feature_matrix = self.tokenizer.texts_to_matrix([feature])
        # boots  [[0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
        # harrds [[0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]

        return self.scaler.transform(feature_matrix)

    def __transform_output(self, prediction):

        predicted_index = np.argmax(prediction)
        print("max arg: {0}".format(predicted_index))
        print(self.encoder.classes_)
        prediction_label = self.encoder.classes_[predicted_index]
        return prediction_label

    def __load_model(self):
        try:
            model_path = self.__get_file_path(MODEL_NAME)
            model = keras.load_model(model_path)
            info_message = "Successfully loaded model from path {0} ]".format(model_path)
            logging.info(info_message)
            return model
        except Exception:
            error_message = "Unable to load model from path {0} ]".format(model_path)
            logging.exception(error_message)
            raise ModelException(error_message)

    def __load_scaler(self):
        try:
            scaler_path = self.__get_file_path(SCALER_NAME)
            info_message = "Successfully loaded scaler from path {0}".format(scaler_path)
            logging.info(info_message)
            print(scaler_path)
            return pickle.load(open(scaler_path, 'rb'))
        except Exception:
            error_message = "Unable to load scaler from path {0}".format(scaler_path)
            logging.exception(error_message)
            raise ModelException(error_message)

    def __load_tokenizer(self):
        try:
            tokenizer_path = self.__get_file_path(TOKENIZER_NAME)
            info_message = "Successfully loaded tokenizer from path {0}".format(tokenizer_path)
            logging.info(info_message)
            print(tokenizer_path)
            return pickle.load(open(tokenizer_path, 'rb'))
        except Exception:
            error_message = "Unable to load tokenizer from path {0}".format(tokenizer_path)
            logging.exception(error_message)
            raise ModelException(error_message)

    def __load_encoder(self):
        try:
            encoder_path = self.__get_file_path(ENCODER_NAME)
            info_message = "Successfully loaded encoder from path {0}".format(encoder_path)
            logging.info(info_message)
            print(encoder_path)
            return pickle.load(open(encoder_path, 'rb'))
        except Exception:
            error_message = "Unable to load scaler from path {0}".format(encoder_path)
            logging.exception(error_message)
            raise ModelException(error_message)

    def __get_file_path(self, file_name):
        return '{0}/{1}'.format(self.resource_path, file_name)

    def __str__(self):
        return str(self.model.summary())


class ModelException(Exception):
    pass
