from absl import app
from absl import flags

import tensorflow as tf

from flags import define_flags


def main(_):
    (train_images,
     train_label), (test_images,
                    test_labels) = tf.keras.datasets.mnist.load_data()


if __name__ == '__main__':
    define_flags()
    app.run(main)
