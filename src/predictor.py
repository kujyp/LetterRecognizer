import time
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from train_model import fill_feed_dict, do_eval
from train_model import placeholder_inputs, graph_model, calcul_loss, training, evaluation

def predict(img, HYPARMS):
    sess = tf.Session()
    ckpt = tf.train.get_checkpoint_state(HYPARMS.ckpt_dir)
    saver = tf.train.Saver()
    saver.restore(sess, ckpt.model_checkpoint_path)

    placebundle = placeholder_inputs(1)
    logits = graph_model(placebundle)
    output = tf.argmax(logits,1)
    # loss = calcul_loss(logits, placebundle)
    # train_op = training(loss, HYPARMS.learning_rate)
    # eval_correct = evaluation(logits, placebundle)
    return sess.run(output, feed_dict = {placebundle.x: img,
                                  placebundle.y_: 0,
                                  placebundle.keep_prob: 1})