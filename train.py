

import os
import shutil
import GetData 
import tensorflow as tf
from model import Model
import _pickle as pickle
from numpy import newaxis
from preprocess import get_random_wav,spec_to_batch
from preprocess import to_spectrogram, get_magnitude
from config import ModelConfig,TrainConfig,Diff,CONFIG_MAP



import matplotlib as plt
import librosa.display

# TODO multi-gpu
def train():

    dsd_train, dsd_test = GetData.getDSDFilelist("DSD100.xml")

    dataset = dict()
    dataset["train_sup"] = dsd_train # 50 training tracks from DSD100 as supervised dataset
    dataset["valid"] =  dsd_test[:25]
    dataset["test"] = dsd_test[25:]

    with open('dataset.pkl', 'wb') as file:
        pickle.dump(dataset, file)
        print("Created dataset structure")
    # Model
    model = Model()

    # Loss, Optimizer
    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
    loss_fn = model.loss()
    lr = ((CONFIG_MAP['flat-R-VAE'].hparams.learning_rate - CONFIG_MAP['flat-R-VAE'].hparams.min_learning_rate) *
        tf.pow(CONFIG_MAP['flat-R-VAE'].hparams.decay_rate, tf.to_float(self.global_step)) +CONFIG_MAP['flat-R-VAE'].hparams.min_learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss_fn, global_step=global_step)

    # Summaries
    summary_op = summaries(model, loss_fn)

    with tf.Session(config=TrainConfig.session_conf) as sess:

        # Initialized, Load state
        sess.run(tf.global_variables_initializer())
        #model.load_state(sess, TrainConfig.CKPT_PATH)

        writer = tf.summary.FileWriter(TrainConfig.GRAPH_PATH, sess.graph)

        # Input source
        
        btch_size = CONFIG_MAP['flat-R-VAE'].hparams.batch_size
        loss = Diff()
        i=0;
        for step in range(global_step.eval(), TrainConfig.FINAL_STEP): # changed xrange to range for py3
            if(i>50)
                i=0;
            batch_ =dsd_train[i:i+btch_size]
            i =i+btch_size
            mixes_wav,drums_wav = get_random_wav(batch_, TrainConfig.SECONDS, ModelConfig.SR)


            mixed_spec = to_spectrogram(mixed_wav)
            mixed_mag = get_magnitude(mixed_spec)

            drums_spec = to_spectrogram(drums_wav)
            drums_mag = get_magnitude(drums_spec)

            

            mixed_batch, _ = model.spec_to_batch(mixed_mag)
            drums_batch, _ = model.spec_to_batch(drums_mag)
           

            l, _, summary = sess.run([loss_fn, optimizer, summary_op],
                                     feed_dict={model.x_mixed: mixed_batch, model.x_drums :drums_batch, model.y_drums: drums_batch})

            loss.update(l)
            print('step-{}\td_loss={:2.2f}\tloss={}'.format(step, loss.diff * 100, loss.value))

            writer.add_summary(summary, global_step=step)

            # Save state
            if step % TrainConfig.CKPT_STEP == 0:
                tf.train.Saver().save(sess, TrainConfig.CKPT_PATH + '/checkpoint', global_step=step)

        writer.close()


def summaries(model, loss):
    for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        tf.summary.histogram(v.name, v)
        tf.summary.histogram('grad/' + v.name, tf.gradients(loss, v))
    tf.summary.scalar('loss', loss)
    tf.summary.histogram('x_mixed', model.x_mixed)
    tf.summary.histogram('x_drums', model.x_drums)
    tf.summary.histogram('y_drums', model.y_drums)
    return tf.summary.merge_all()


def setup_path():
    if TrainConfig.RE_TRAIN:
        if os.path.exists(TrainConfig.CKPT_PATH):
            shutil.rmtree(TrainConfig.CKPT_PATH)
        if os.path.exists(TrainConfig.GRAPH_PATH):
            shutil.rmtree(TrainConfig.GRAPH_PATH)
    if not os.path.exists(TrainConfig.CKPT_PATH):
        os.makedirs(TrainConfig.CKPT_PATH)


if __name__ == '__main__':
    setup_path()
    train()