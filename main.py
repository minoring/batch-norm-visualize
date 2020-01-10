from datetime import datetime

from absl import app
from absl import flags
import tensorflow as tf
import numpy as np

from flags import define_flags
from model import mnist_model
from model import mnist_model_bn


def preprocess(input_img, label):
  return (tf.cast(input_img, tf.float32) / 127.5 - 1.0), label


@tf.function
def train_step(model, optimizer, loss_fn, training, input_img, label):
  with tf.GradientTape() as tape:
    output1, output2, output3, pred = model(input_img, training=training)
    loss = loss_fn(label, pred)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  return output1, output2, output3, loss


def main(_):
  FLAGS = flags.FLAGS

  # MNIST dataset only for now.
  (train_images,
   train_labels), (test_images,
                   test_labels) = tf.keras.datasets.mnist.load_data()

  train_dataset = tf.data.Dataset.from_tensor_slices(
      (train_images, train_labels))
  train_dataset = train_dataset.repeat().shuffle(10000).batch(60).map(
      preprocess)

  test_images = np.apply_along_axis(lambda x: x / 127.5 - 1.0,
                                    axis=0,
                                    arr=test_images)

  if FLAGS.bn:
    model = mnist_model_bn()
    logdir = './logs/scalars/' + 'batch_norm'
  else:
    model = mnist_model()
    logdir = './logs/scalars/' + 'without_batch_norm'

  optimizer = tf.keras.optimizers.SGD()
  loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

  model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
  summary_writer = tf.summary.create_file_writer(logdir)

  for epoch in range(FLAGS.epochs):
    for dataset in train_dataset:
      output1, output2, output3, loss = train_step(model,
                                                   optimizer=optimizer,
                                                   loss_fn=loss_fn,
                                                   training=True,
                                                   input_img=dataset[0],
                                                   label=dataset[1])
    print('epoch: {} loss: {:.4f}'.format(epoch, loss))
    _, _, _, pred = model.predict(test_images)
    prediction = tf.math.argmax(pred, axis=1)
    acc = np.sum(test_labels == prediction) / test_labels.shape[0]

    with summary_writer.as_default():
      tf.summary.scalar('accuracy', data=acc, step=epoch)
      tf.summary.scalar(name='output1_mean',
                        data=tf.math.reduce_mean(output1),
                        step=epoch)
      tf.summary.scalar(name='output1_std',
                        data=tf.math.reduce_std(output1),
                        step=epoch)
      tf.summary.scalar(name='output2_mean',
                        data=tf.math.reduce_mean(output2),
                        step=epoch)
      tf.summary.scalar(name='output2_std',
                        data=tf.math.reduce_std(output2),
                        step=epoch)
      tf.summary.scalar(name='output3_mean',
                        data=tf.math.reduce_mean(output3),
                        step=epoch)
      tf.summary.scalar(name='output3_std',
                        data=tf.math.reduce_std(output3),
                        step=epoch)


if __name__ == '__main__':
  define_flags()
  app.run(main)
