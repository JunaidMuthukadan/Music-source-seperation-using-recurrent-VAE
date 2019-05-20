# Musical Source Separation using Deep Recurrent Variational Autoencoder

>* [__Take a look at the demo!__](https://www.youtube.com/)

## Intro
Traditionally, discriminative training for source separation is proposed using deep neural networks or non-negative matrix factorization. In this project i proposed a variational autoencoder based framework using recurrent neural networks for blind musical source (bass,drums ,vocals and others) separation. It is principled generative model instead of traditional discriminative models


## Implementations
* I used Recurrent neural network for both encoder and decoder
  * 3 RNN layers + 2 dense layer + 2 time-frequency masking layer
* I used DSD100 [__Dataset__] (https://sigsep.github.io/datasets/dsd100.html) for training my model

## Requirements
* Numpy >= 1.3.0
* TensorFlow == 1.2
* librosa == 0.5.1

## Usage
* Configuration
  * config.py: set dataset path appropriately.
* Training
  * ```python train.py```
  * check the loss graph in Tensorboard.
* Evaluation
  * ``` python eval.py```
  * check the result in Tensorboard (audio tab).

