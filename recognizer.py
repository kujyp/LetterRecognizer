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
from src.predictor import predict

class Recognizer:
    def __init__(self,HYPARMS):
        self.HYPARMS = HYPARMS
    def run(self,input):
        input = self.preprocess(input)
        return self.getPredict(input)

    def preprocess(self, input):
        input = self.resize(input)
        return input

    def resize(self, input):
        return input

    def getPredict(self, input):
        return predict(input, self.HYPARMS)

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
    return numpy.array(img)

def main(_):
    HYPARMS = load_params()
    if tf.gfile.Exists(HYPARMS.log_dir):
        tf.gfile.DeleteRecursively(HYPARMS.log_dir)
    tf.gfile.MakeDirs(HYPARMS.log_dir)

    input = load_image(os.path.join('input', 'mnist.jpeg'))
    print("Answer : " + Recognizer(HYPARMS).run(input))
    # while(True):
    #     filename = 'mnist.jpeg'
    #     if(False):
    #         break
    #     else:
    #         input = load_image(os.path.join('input',filename))
    #         print("Answer : " + Recognizer().run(input))

if __name__ == "__main__":
    tf.app.run(main=main)