# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import math
import sys
import os
import numpy
from PIL import Image

from src.params import Hyparms
from src.train_runner import run_training
from src.predictor import Predictor

class Recognizer:
    def __init__(self,predictor):
        self.predictor = predictor
    def run(self,input):
        input = self.preprocess(input)
        return self.getPredict(input)

    def preprocess(self, input):
        input = input.resize([28,28])
        input = input.convert('L')
        input = numpy.array(input)
        input = input.reshape([1, 784])
        return input

    def resize(self, input):
        input.resize([28,28])
        return input

    def getPredict(self, input):
        return self.predictor.predict(input)


def allfiles(path):
    res = []

    for root, dirs, files in os.walk(path):
        for file in files:
            filepath = os.path.join(os.path.abspath(path), file)
            res.append(filepath)

    return res

def load_params():
    HYPARMS = Hyparms()

    # Set parameters
    #HYPARMS.max_steps = 200
    #HYPARMS.learning_rate = 0.1
    #HYPARMS.batch_size = 128
    #HYPARMS.log_dir = "logs"
    #HYPARMS.dropout_rate = 0.8
    #HYPARMS.input_data_dir = 'MNIST_data'

    return HYPARMS

def load_image(filename):
    img = Image.open(filename)
    return img

def print_prediction(file, predictor):
    input = load_image(file)
    print("File : ")
    print(file)
    print("Answer : ")
    print(Recognizer(predictor).run(input))


def main(_):
    HYPARMS = load_params()
    if tf.gfile.Exists(HYPARMS.log_dir):
        tf.gfile.DeleteRecursively(HYPARMS.log_dir)
    tf.gfile.MakeDirs(HYPARMS.log_dir)

    predictor = Predictor(HYPARMS)

    files = allfiles(HYPARMS.input_data_dir)
    for file in files:
        print_prediction(file, predictor)

if __name__ == "__main__":
    tf.app.run(main=main)

