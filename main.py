from datetime import datetime

from absl import app
from absl import flags
import tensorflow as tf

from flags import define_flags
from model import mnist_model
from model import mnist_model_bn


def preprocess(input_image, label):
  return (tf.cast(input_image, tf.float32) / 127.5 - 1.0), label


def main(_):
  FLAGS = flags.FLAGS

  # MNIST dataset only for now.
  (train_images,
   train_labels), (test_images,
                   test_labels) = tf.keras.datasets.mnist.load_data()

  train_dataset = tf.data.Dataset.from_tensor_slices(
      (train_images, train_labels))
  test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
  train_dataset = train_dataset.repeat().shuffle(10000).batch(60).map(
      preprocess)
  test_dataset = test_dataset.batch(60).map(preprocess)

  if FLAGS.bn:
    model = mnist_model_bn()
    logdir = './logs/scalars/' + 'batch_norm'
  else:
    model = mnist_model()
    logdir = './logs/scalars/' + 'without_batch_norm'

  model.compile(optimizer=tf.keras.optimizers.SGD(), # Default learning rate 0.01
                loss='categorical_crossentropy',
                metrics=['accuracy'])

  
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

  model.fit(train_dataset,
            epochs=1,
            steps_per_epoch=600000,
            callbacks=[tensorboard_callback])

if __name__ == '__main__':
  define_flags()
  app.run(main)
