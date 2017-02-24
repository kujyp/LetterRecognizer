import time
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from train_model import fill_feed_dict, do_eval
from train_model import placeholder_inputs, graph_model, calcul_loss, training, evaluation

saverflag = False

class Predictor:
    sess = tf.Session()
    def __init__(self,HYPARMS):
        self.init(HYPARMS)

    def predict(self, img):
        return self.sess.run(self.output, feed_dict = {self.placebundle.x: img,
                                  self.placebundle.keep_prob: 1})

    def init(self,HYPARMS):
        sess = tf.Session()
        self.placebundle = placeholder_inputs(1)

        self.init = tf.initialize_all_variables()
        self.saver = tf.train.Saver()
        self.sess.run(self.init)

        self.ckpt = tf.train.get_checkpoint_state(HYPARMS.ckpt_dir)
        self.saver.restore(sess, self.ckpt.model_checkpoint_path)

        self.logits = graph_model(self.placebundle)
        self.output = tf.argmax(self.logits,1)

