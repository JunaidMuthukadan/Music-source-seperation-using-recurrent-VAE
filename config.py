# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 13:57:59 2019

@author: JuNaiD
"""
import tensorflow as tf
import collections
from tensorflow.contrib.training import HParams



class Config(collections.namedtuple(
    'Config',
    ['hparams',
     'train_examples_path', 'eval_examples_path'])):

  def values(self):
    return self._asdict()


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





class ModelConfig:
    SR = 16000                # Sample Rate
    L_FRAME = 1024            # default 1024
    L_HOP = closest_power_of_two(L_FRAME / 4)
    SEQ_LEN = 252
    
    
class TrainConfig:
    LR = 0.0001
    GRAPH_PATH = 'graphs/'
    CKPT_PATH = 'checkpoints/'
    FINAL_STEP = 100000
    CKPT_STEP = 500
    SECONDS = 4 # To get 512,512 in melspecto
    RE_TRAIN = True
    session_conf = tf.ConfigProto(
        device_count={'CPU': 1, 'GPU': 1},
        gpu_options=tf.GPUOptions(
            allow_growth=True,
            per_process_gpu_memory_fraction=0.25
        ),
    )

def merge_hparams(hparams_1, hparams_2):
  """Merge hyperparameters from two tf.contrib.training.HParams objects.
  If the same key is present in both HParams objects, the value from `hparams_2`
  will be used.
  Args:
    hparams_1: The first tf.contrib.training.HParams object to merge.
    hparams_2: The second tf.contrib.training.HParams object to merge.
  Returns:
    A merged tf.contrib.training.HParams object with the hyperparameters from
    both `hparams_1` and `hparams_2`.
  """
  hparams_map = hparams_1.values()
  hparams_map.update(hparams_2.values())
  return tf.contrib.training.HParams(**hparams_map)

def get_default_hparams():
  return tf.contrib.training.HParams(
      #max_seq_len=32,  # Maximum sequence length. Others will be truncated.
      z_size=256,  # Size of latent vector z.
      free_bits=0.0,  # Bits to exclude from KL loss per dimension.
      max_beta=1.0,  # Maximum KL cost weight, or cost if not annealing.
      beta_rate=0.0,  # Exponential rate at which to anneal KL cost.
      batch_size=8,  # Minibatch size.
      grad_clip=1.0,  # Gradient clipping. Recommend leaving at 1.0.
      clip_mode='global_norm',  # value or global_norm.
      # If clip_mode=global_norm and global_norm is greater than this value,
      # the gradient will be clipped to 0, effectively ignoring the step.
      dropout_keep_prob = 1.0,  # Probability all dropout keep.
      sampling_schedule ='constant',  # constant, exponential, inverse_sigmoid
      sampling_rate =0.0,  # Interpretation is based on `sampling_schedule`.
      residual_encoder = False,  # Use residual connections in encoder.
      residual_decoder =False,  # Use residual connections in decoder.
      grad_norm_clip_to_zero=10000,
      learning_rate=0.001,  # Learning rate.
      decay_rate=0.9999,  # Learning rate decay per minibatch.
      min_learning_rate=0.00001,  # Minimum learning rate.
  )


CONFIG_MAP = {}

CONFIG_MAP['flat-R-VAE'] = Config(
    hparams=merge_hparams(
        get_default_hparams(),
        HParams(
            #max_seq_len=256,
            enc_rnn_size=[2048, 2048],
            dec_rnn_size=[2048, 2048, 2048],
            #free_bits=256,
            max_beta=0.2,
        )),
    #data_converter=mel_16bar_converter,
    train_examples_path=None,
    eval_examples_path=None,
)

