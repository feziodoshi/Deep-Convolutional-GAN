# Deep-Convolutional-GAN

This is a tensorflow implementation of the paper titled ** Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks**-https://arxiv.org/abs/1511.06434

# Prerequisites:

python
tensorflow
numpy
matplotlib

To run the training code, can simply run the code: first.py

# Intuitive understanding of GAN

The model has two parts:
Generator and Discriminator.
Generative Networks are learnt to generate pics/ data that looks the same- that is they model data(fake data) from random noise which is identical/ seems to have been sampled from regular training data

Discriminiative Networks are learnt to discriminate between and check if the generated data is from the distribution of training data or from newly generated image- kind of finding if it is fake or not.

Thus over time generative networks become good at cheating discriminative networks and discriminiative networks become good at finding if the data is fake or not

