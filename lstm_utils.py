# -*- coding: utf-8 -*-
"""
Created on Sun May  5 16:18:24 2019

@author: JuNaiD
"""

# Copyright 2019 The Magenta Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MusicVAE LSTM model utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import seq2seq
from tensorflow.contrib.cudnn_rnn.python.layers import cudnn_rnn
from tensorflow.python.util import nest


def rnn_cell(rnn_cell_size, dropout_keep_prob, residual, is_training=True):
  """Builds an LSTMBlockCell based on the given parameters."""
  dropout_keep_prob = dropout_keep_prob if is_training else 1.0
  cells = []
  for i in range(len(rnn_cell_size)):
    cell = rnn.LSTMBlockCell(rnn_cell_size[i])
    if residual:
      cell = rnn.ResidualWrapper(cell)
      if i == 0 or rnn_cell_size[i] != rnn_cell_size[i - 1]:
        cell = rnn.InputProjectionWrapper(cell, rnn_cell_size[i])
    cell = rnn.DropoutWrapper(
        cell,
        input_keep_prob=dropout_keep_prob)
    cells.append(cell)
  return rnn.MultiRNNCell(cells)



def initial_cell_state_from_embedding(cell, z, name=None):
  """Computes an initial RNN `cell` state from an embedding, `z`."""
  flat_state_sizes = nest.flatten(cell.state_size)
  return nest.pack_sequence_as(
      cell.zero_state(batch_size=z.shape[0], dtype=tf.float32),
      tf.split(
          tf.layers.dense(
              z,
              sum(flat_state_sizes),
              activation=tf.tanh,
              kernel_initializer=tf.random_normal_initializer(stddev=0.001),
              name=name),
          flat_state_sizes,
          axis=1))



class LstmDecodeResults(
    collections.namedtuple('LstmDecodeResults',
                           ('rnn_input', 'rnn_output', 'samples', 'final_state',
                            'final_sequence_lengths'))):
  pass


class Seq2SeqLstmDecoderOutput(
    collections.namedtuple('BasicDecoderOutput',
                           ('rnn_input', 'rnn_output', 'sample_id'))):
  pass


class Seq2SeqLstmDecoder(seq2seq.BasicDecoder):
  """Overrides BaseDecoder to include rnn inputs in the output."""

  def __init__(self, cell, helper, initial_state, input_shape,
               output_layer=None):
    self._input_shape = input_shape
    super(Seq2SeqLstmDecoder, self).__init__(
        cell, helper, initial_state, output_layer)

  @property
  def output_size(self):
    return Seq2SeqLstmDecoderOutput(
        rnn_input=self._input_shape,
        rnn_output=self._rnn_output_size(),
        sample_id=self._helper.sample_ids_shape)

  @property
  def output_dtype(self):
    dtype = nest.flatten(self._initial_state)[0].dtype
    return Seq2SeqLstmDecoderOutput(
        dtype,
        nest.map_structure(lambda _: dtype, self._rnn_output_size()),
        self._helper.sample_ids_dtype)

  def step(self, time, inputs, state, name=None):
    results = super(Seq2SeqLstmDecoder, self).step(time, inputs, state, name)
    outputs = Seq2SeqLstmDecoderOutput(
        rnn_input=inputs,
        rnn_output=results[0].rnn_output,
        sample_id=results[0].sample_id)
    return (outputs,) + results[1:]


