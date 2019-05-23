# Musical Source Separation using Deep Recurrent Variational Autoencoder

>* [__Take a look at the demo!__](https://www.youtube.com/)

## Intro
Traditionally, discriminative training for musical source separation is proposed using deep neural networks or non-negative matrix factorization. In this project i proposed a variational autoencoder (VAE) based framework using recurrent neural networks for blind musical source (bass,drums ,vocals and others) separation. It is principled generative model compared to other traditional discriminative models.

<img src="img/vae.png">


## What is Recurrent VAE for Music Source Separation?
  The Recurrent VAE for Music Source Separation is a recurrent neural network variational autoencoder decder architecture that operates on magnitude spectrum of the mixed music and generates seperated musical sourses (bass,drums, vocals and etc).

  <img src="img/brief.png">

  The encoder structure is actually taken from goolge's Music-VAE architecture,It uses a two-layer bidirectional
  LSTM network (Hochreiter & Schmidhuber, 1997; Schuster & Paliwal, 1997). It process an input music sepectogram 
  x = {x1, x2, . . . , xL } to obtain the final state vectors from the second bidirectional LSTM layer. These
  are then concatenated to produce hT and fed into two fullyconnected layers to produce the latent distribution parameters µ and σ.

  The decoder structure is, instead of Hierarchical decoder model as it is in google's Music-VAE I uses a simple uni-directinal two-layer LSTM network with tensorflows seq2seq, to test the performance.


  <img src="img/structure.png">

* I used DSD100 [__Dataset__] (https://sigsep.github.io/datasets/dsd100.html) for training my model

## Requirements
* Numpy >= 1.3.0
* TensorFlow == 1.2
* librosa == 0.5.1

## Usage
* Training
  * ```python train.py```
  * check the loss graph in Tensorboard.


