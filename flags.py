from absl import flags


def define_flags():
  flags.DEFINE_string('dataset', 'mnist', 'Dataset to experiment [mnist]')
  flags.DEFINE_float('learning_rate', 0.1, 'Learning rate')
  flags.DEFINE_boolean('bn', False, 'Use Batch normalization')
  flags.DEFINE_integer('epochs', 100, 'Number of epochs')
  flags.DEFINE_integer('steps_per_epoch', 600, 'Steps per epoch')
  flags.DEFINE_integer('validation_steps', 10, 'Validation steps')
