from __future__ import division
import tensorflow as tf
import numpy as np

class Diff(object):
    def __init__(self, v=0.):
        self.value = v
        self.diff = 0.

    def update(self, v):
        if self.value:
            diff = (v / self.value - 1)
            self.diff = diff
        self.value = v


def shape(tensor):
    s = tensor.get_shape()
    return tuple([s[i].value for i in range(0, len(s))])


# TODO general pretty print
def pretty_list(list):
    return ', '.join(list)

def pretty_dict(dict):
    return '\n'.join('{} : {}'.format(k, v) for k, v in dict.items())

def closest_power_of_two(target):
    if target > 1:
        for i in range(1, int(target)):
            if (2 ** i >= target):
                pwr = 2 ** i
                break
        if abs(pwr - target) < abs(pwr/2 - target):
            return pwr
        else:
            return int(pwr / 2)
    else:
        return 1

# Write the nd array to txtfile
def nd_array_to_txt(filename, data):
    path = filename + '.txt'
    file = open(path, 'w')
    with file as outfile:
        # I'm writing a header here just for the sake of readability
        # Any line starting with "#" will be ignored by numpy.loadtxt
        outfile.write('# Array shape: {0}\n'.format(data.shape))

        # Iterating through a ndimensional array produces slices along
        # the last axis. This is equivalent to data[i,:,:] in this case
        for data_slice in data:

            # The formatting string indicates that I'm writing out
            # the values in left-justified columns 7 characters in width
            # with 2 decimal places.
            np.savetxt(outfile, data_slice, fmt='%-7.2f')

            # Writing out a break to indicate different slices...
            outfile.write('# New slice\n')


def flatten_maybe_padded_sequences(maybe_padded_sequences, lengths=None):
  """Flattens the batch of sequences, removing padding (if applicable).
  Args:
    maybe_padded_sequences: A tensor of possibly padded sequences to flatten,
        sized `[N, M, ...]` where M = max(lengths).
    lengths: Optional length of each sequence, sized `[N]`. If None, assumes no
        padding.
  Returns:
     flatten_maybe_padded_sequences: The flattened sequence tensor, sized
         `[sum(lengths), ...]`.
  """
  def flatten_unpadded_sequences():
    # The sequences are equal length, so we should just flatten over the first
    # two dimensions.
    return tf.reshape(maybe_padded_sequences,
                      [-1] + maybe_padded_sequences.shape.as_list()[2:])

  if lengths is None:
    return flatten_unpadded_sequences()

  def flatten_padded_sequences():
    indices = tf.where(tf.sequence_mask(lengths))
    return tf.gather_nd(maybe_padded_sequences, indices)

  return tf.cond(
      tf.equal(tf.reduce_min(lengths), tf.shape(maybe_padded_sequences)[1]),
      flatten_unpadded_sequences,
      flatten_padded_sequences)