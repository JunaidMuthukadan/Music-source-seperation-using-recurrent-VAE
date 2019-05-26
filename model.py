from __future__ import division	
import os
import lstm_utils
import numpy as np
import tensorflow as tf
from utils import shape
from tensorflow.contrib import rnn
import tensorflow_probability as tfp
from config import ModelConfig,CONFIG_MAP
from tensorflow.contrib.rnn import GRUCell, MultiRNNCell
from tensorflow.python.layers import core as layers_core


seq2seq = tf.contrib.seq2seq
ds = tfp.distributions


class Model:
    def __init__(self):

        # Input, Output
        self.x_mixed = tf.placeholder(tf.float32, shape=(None, None, ModelConfig.L_FRAME // 2 + 1), name='x_mixed')
        #self.y_vocals = tf.placeholder(tf.float32, shape=(None, None, ModelConfig.L_FRAME // 2 + 1), name='y_src1')
        #self.y_bass = tf.placeholder(tf.float32, shape=(None, None, ModelConfig.L_FRAME // 2 + 1), name='y_src2')
        self.x_drums = tf.placeholder(tf.float32, shape=(None, None, ModelConfig.L_FRAME // 2 + 1), name='x_drums')
        self.y_drums = tf.placeholder(tf.float32, shape=(None, None, ModelConfig.L_FRAME // 2 + 1), name='y_drums')
        #self.y_other = tf.placeholder(tf.float32, shape=(None, None, ModelConfig.L_FRAME // 2 + 1), name='y_src4')




        # Network
        
        self.dropout_keep_prob =1
        #self.residual
        self.net = tf.make_template('net', self._net)
        self()

    def __call__(self):
        return self.net()

    def _net(self):
        # RNN and dense layers
        cells_fw = []
        cells_bw = []
        for i, layer_size in enumerate(CONFIG_MAP['flat-R-VAE'].hparams.enc_rnn_size):
           cells_fw.append(
                lstm_utils.rnn_cell(
                    [layer_size], CONFIG_MAP['flat-R-VAE'].hparams.dropout_keep_prob,
                    CONFIG_MAP['flat-R-VAE'].hparams.residual_encoder))
           cells_bw.append(
                lstm_utils.rnn_cell(
                    [layer_size], CONFIG_MAP['flat-R-VAE'].hparams.dropout_keep_prob,
                    CONFIG_MAP['flat-R-VAE'].hparams.residual_encoder))
    

        _, states_fw, states_bw = rnn.stack_bidirectional_dynamic_rnn(cells_fw,cells_bw,self.x_mixed,dtype=tf.float32)
        # Note we access the outputs (h) from the states since the backward
        # ouputs are reversed to the input order in the returned outputs.
        last_h_fw = states_fw[-1][-1].h
        last_h_bw = states_bw[-1][-1].h      
        last_h =tf.concat([last_h_fw, last_h_bw], 1)   

        mu = tf.layers.dense(
        last_h,
        CONFIG_MAP['flat-R-VAE'].hparams.z_size,
        name='encoder/mu',
        kernel_initializer=tf.random_normal_initializer(stddev=0.001))

        sigma = tf.layers.dense(
        last_h,
        CONFIG_MAP['flat-R-VAE'].hparams.z_size,
        activation=tf.nn.softplus,
        name='encoder/sigma',
        kernel_initializer=tf.random_normal_initializer(stddev=0.001)) 

        q_z =ds.MultivariateNormalDiag(loc=mu, scale_diag=sigma) 

        z = q_z.sample()
        repeated_z = tf.tile(tf.expand_dims(z, axis=1), [1, tf.shape(self.x_drums)[1], 1])

        #p_z = ds.MultivariateNormalDiag(loc=[0.] * hparams.z_size, scale_diag=[1.] * hparams.z_size)



        '''hier_cells = [lstm_utils.rnn_cell(
            hparams.dec_rnn_size,
            dropout_keep_prob=hparams.dropout_keep_prob,
            residual=hparams.residual_decoder)
        for _ in range(len(level_lengths))]'''

        dec_cell = lstm_utils.rnn_cell(
        CONFIG_MAP['flat-R-VAE'].hparams.dec_rnn_size, CONFIG_MAP['flat-R-VAE'].hparams.dropout_keep_prob,
        CONFIG_MAP['flat-R-VAE'].hparams.residual_decoder, True)

        x_input = tf.concat([self.x_drums, repeated_z], axis=2)
        x_length = shape(self.x_mixed)[2]
        x_length_b = np.array(CONFIG_MAP['flat-R-VAE'].hparams.batch_size*[x_length])
        helper = seq2seq.TrainingHelper(x_input, x_length_b)
        output_layer = layers_core.Dense(x_length, name='output_projection')
        initial_state = lstm_utils.initial_cell_state_from_embedding(dec_cell, z, name='decoder/z_to_initial_state')

        decoder = lstm_utils.Seq2SeqLstmDecoder(dec_cell,helper,initial_state=initial_state,input_shape=helper.inputs.shape[2:],output_layer=output_layer)
        max_length =None # (Optional) The maximum iterations to decode.
        final_output, final_state, final_lengths = seq2seq.dynamic_decode(decoder,maximum_iterations=max_length,swap_memory=True,scope='decoder')
        #flat_x_target = flatten_maybe_padded_sequences(x_target, x_length)
        flat_rnn_output = flatten_maybe_padded_sequences(final_output.rnn_output, x_length_b)
        
        
        
        
        '''output_rnn, rnn_state = tf.nn.dynamic_rnn(rnn_layer, self.x_mixed, dtype=tf.float32)
        
        y_hat_src1 = tf.layers.dense(inputs=output_rnn, units=input_size, activation=tf.nn.relu, name='y_hat_src1')
        y_hat_src2 = tf.layers.dense(inputs=output_rnn, units=input_size, activation=tf.nn.relu, name='y_hat_src2')

        # time-freq masking layer
        y_tilde_src1 = y_hat_src1 / (y_hat_src1 + y_hat_src2 + np.finfo(float).eps) * self.x_mixed
        y_tilde_src2 = y_hat_src2 / (y_hat_src1 + y_hat_src2 + np.finfo(float).eps) * self.x_mixed

        return y_tilde_src1, y_tilde_src2'''
        return flat_rnn_output




    def _flat_reconstruction_loss(self, flat_x_target, flat_rnn_output):
        flat_logits = flat_rnn_output
        flat_truth = tf.argmax(flat_x_target, axis=1)
        flat_predictions = tf.argmax(flat_logits, axis=1)
        r_loss = tf.nn.softmax_cross_entropy_with_logits(labels=flat_x_target, logits=flat_logits)

        metric_map = {
        'metrics/accuracy':
            tf.metrics.accuracy(flat_truth, flat_predictions),
        'metrics/mean_per_class_accuracy':
            tf.metrics.mean_per_class_accuracy(
                flat_truth, flat_predictions, flat_x_target.shape[-1].value),
        }
        return r_loss, metric_map




    def loss(self):
        flat_rnn_output = self()
        p_z = ds.MultivariateNormalDiag(loc=[0.] * CONFIG_MAP['flat-R-VAE'].hparams.z_size, scale_diag=[1.] * CONFIG_MAP['flat-R-VAE'].hparams.z_size)
        kl_div = ds.kl_divergence(q_z, p_z)
        flat_x_target = flatten_maybe_padded_sequences(y_drums, x_length_b)
        r_loss, metric_map = self._flat_reconstruction_loss(flat_x_target, flat_rnn_output)
        cum_x_len = tf.concat([(0,), tf.cumsum(x_length_b)], axis=0)
        r_losses = []
        for i in range(CONFIG_MAP['flat-R-VAE'].hparams.batch_size):
            b, e = cum_x_len[i], cum_x_len[i + 1]
            r_losses.append(tf.reduce_sum(r_loss[b:e]))
        r_loss = tf.stack(r_losses)
        
        free_nats = CONFIG_MAP['flat-R-VAE'].hparams.free_bits * tf.math.log(2.0)
        kl_cost = tf.maximum(kl_div - free_nats, 0)
        beta = ((1.0 - tf.pow(CONFIG_MAP['flat-R-VAE'].hparams.beta_rate, tf.to_float(self.global_step)))* CONFIG_MAP['flat-R-VAE'].hparams.max_beta)
        return tf.reduce_mean(r_loss) + beta * tf.reduce_mean(kl_cost)


        
    @staticmethod
    def spec_to_batch(src):
        num_wavs, freq, n_frames = src.shape

        # Padding
        pad_len = 0
        if n_frames % ModelConfig.SEQ_LEN > 0:
            pad_len = (ModelConfig.SEQ_LEN - (n_frames % ModelConfig.SEQ_LEN))
        pad_width = ((0, 0), (0, 0), (0, pad_len))
        padded_src = np.pad(src, pad_width=pad_width, mode='constant', constant_values=0)

        assert(padded_src.shape[-1] % ModelConfig.SEQ_LEN == 0)

        batch = np.reshape(padded_src.transpose(0, 2, 1), (-1, ModelConfig.SEQ_LEN, freq))
        return batch, padded_src

    @staticmethod
    def load_state(sess, ckpt_path):
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(ckpt_path + '/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            tf.train.Saver().restore(sess, ckpt.model_checkpoint_path)
