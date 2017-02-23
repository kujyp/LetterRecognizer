from src.params import Hyparms
from src.train_runner import run_training
import tensorflow as tf

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

def main(_):
    HYPARMS = load_params()
    if tf.gfile.Exists(HYPARMS.log_dir):
        tf.gfile.DeleteRecursively(HYPARMS.log_dir)
    tf.gfile.MakeDirs(HYPARMS.log_dir)

    tf.gfile.MakeDirs(HYPARMS.ckpt_dir)

    run_training(HYPARMS)

if __name__ == '__main__':
  tf.app.run(main=main)