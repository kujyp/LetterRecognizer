import time
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from train_model import fill_feed_dict, do_eval
from train_model import placeholder_inputs, graph_model, calcul_loss, training, evaluation

def predict(img, HYPARMS):
    placebundle = placeholder_inputs(1)
    ckpt = tf.train.get_checkpoint_state(HYPARMS.ckpt_dir)

    init = tf.initialize_all_variables()
    saver = tf.train.Saver()

    sess = tf.Session()
    sess.run(init)

    saver.restore(sess, ckpt.model_checkpoint_path)

    logits = graph_model(placebundle)
    output = tf.argmax(logits,1)
    # loss = calcul_loss(logits, placebundle)
    # train_op = training(loss, HYPARMS.learning_rate)
    # eval_correct = evaluation(logits, placebundle)
    return sess.run(output, feed_dict = {placebundle.x: img,
                                  placebundle.keep_prob: 1})