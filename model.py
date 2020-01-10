import tensorflow as tf


def mnist_model():
  """Returns mnist model"""
  initializer = tf.random_normal_initializer()
  model = tf.keras.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(100,
                            kernel_initializer=initializer,
                            activation='sigmoid'),
      tf.keras.layers.Dense(100,
                            kernel_initializer=initializer,
                            activation='sigmoid'),
      tf.keras.layers.Dense(100,
                            kernel_initializer=initializer,
                            activation='sigmoid'),
      tf.keras.layers.Dense(10,
                            kernel_initializer=initializer,
                            activation='softmax')
  ])

  return model


def mnist_model_bn():
  """Returns mnist model with batch normalization"""
  initializer = tf.random_normal_initializer()
  model = tf.keras.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(100,
                            kernel_initializer=initializer,
                            activation='sigmoid'),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Dense(100,
                            kernel_initializer=initializer,
                            activation='sigmoid'),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Dense(100,
                            kernel_initializer=initializer,
                            activation='sigmoid'),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Dense(10, kernel_initializer=initializer, activation='softmax')
  ])

  return model
