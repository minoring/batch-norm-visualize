from absl import flags


def define_flags():
    flags.DEFINE_string('dataset', 'mnist', 'Dataset to experiment [mnist]')