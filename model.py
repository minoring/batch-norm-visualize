import tensorflow as tf


def mnist_model():
  """Returns mnist model"""
  initializer = tf.random_normal_initializer()

  inputs = tf.keras.layers.Input(shape=(28, 28))

  x = tf.keras.layers.Flatten(input_shape=(28, 28))(inputs)
  output1 = tf.keras.layers.Dense(100,
                                  kernel_initializer=initializer,
                                  activation='sigmoid')(x)
  output2 = tf.keras.layers.Dense(100,
                                  kernel_initializer=initializer,
                                  activation='sigmoid')(output1)
  output3 = tf.keras.layers.Dense(100,
                                  kernel_initializer=initializer,
                                  activation='sigmoid')(output2)
  pred = tf.keras.layers.Dense(10,
                               kernel_initializer=initializer,
                               activation='softmax')(output3)
  return tf.keras.models.Model(inputs=inputs,
                               outputs=[output1, output2, output3, pred])


def mnist_model_bn():
  """Returns mnist model"""
  initializer = tf.random_normal_initializer()

  inputs = tf.keras.layers.Input(shape=(28, 28))

  x = tf.keras.layers.Flatten(input_shape=(28, 28))(inputs)
  output1 = tf.keras.layers.Dense(100, kernel_initializer=initializer)(x)
  output1 = tf.keras.layers.BatchNormalization()(output1)
  output1 = tf.keras.activations.sigmoid(output1)

  output2 = tf.keras.layers.Dense(100, kernel_initializer=initializer)(output1)
  output2 = tf.keras.layers.BatchNormalization()(output2)
  output2 = tf.keras.activations.sigmoid(output2)

  output3 = tf.keras.layers.Dense(100, kernel_initializer=initializer)(output2)
  output3 = tf.keras.layers.BatchNormalization()(output3)
  output3 = tf.keras.activations.sigmoid(output3)

  pred = tf.keras.layers.Dense(10,
                               kernel_initializer=initializer,
                               activation='softmax')(output3)
  return tf.keras.models.Model(inputs=inputs,
                               outputs=[output1, output2, output3, pred])
