from absl import flags


def define_flags():
  flags.DEFINE_string('dataset', 'mnist', 'Dataset to experiment [mnist]')
  flags.DEFINE_float('learning_rate', 0.1, 'Learning rate')
  flags.DEFINE_boolean('bn', False, 'Use Batch normalization')
