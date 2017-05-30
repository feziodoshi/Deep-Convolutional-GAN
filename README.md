## Deep-Convolutional-GAN

This is a tensorflow implementation of the paper titled __Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks__-https://arxiv.org/abs/1511.06434

## Prerequisites:

* python
* tensorflow
* numpy
* matplotlib

To run the training code, can simply run the code: 
> first.py

## Intuitive understanding of GAN

The model has two parts:
Generator and Discriminator.
Generative Networks are learnt to generate pics/ data that looks the same- that is they model data(fake data) from random noise which is identical/ seems to have been sampled from regular training data

Discriminiative Networks are learnt to discriminate between and check if the generated data is from the distribution of training data or from newly generated image- kind of finding if it is fake or not.

Thus over time generative networks become good at cheating discriminative networks and discriminiative networks become good at finding if the data is fake or not

![Generative Adversarial Network](https://github.com/feziodoshi/Deep-Convolutional-GAN/blob/master/gan_image.png)

The two methods that we will specifically use to optimize our model are
* [Batch Normalization](https://arxiv.org/abs/1502.03167)
* use the activation function Relu

To better optimize stochastically we will be using the __Adam Optimizer__

Now the most important part of training the systems here is making sure some faults do not occur:
1) Discriminator losses become 0 and this will leave no loss for generator to optimize upon
2) Discriminator losses become unbounded and this has no scope for the disrciminator and subsequentially  the generator to improve
3) Divergent Discriminator accuracy

To prevent this we need to run the training on a conditional basis, based on the different  losses.

# Saving the model
Finally the model is saved at regular checkpoints using the Tensorflow Saver Object

# Tensorflow Summary
At every 10th epoch, we will write a summary which includes all the losses:
* Discriminator Real Loss
* Discriminator Fake Loss
* Generator Loss

We will also produce 10 generator image and write its summary.

# Generated output
Using this model we are trying to generate MNIST like dataset. Thus Generated output looks something like

![](https://github.com/feziodoshi/Deep-Convolutional-GAN/blob/master/generated.png)

# References
- https://arxiv.org/abs/1511.06434
- https://github.com/Newmu/dcgan_code
- https://github.com/zsdonghao/dcgan
- [Adversarial PoseNet: A Structure-aware Convolutional Network for Human Pose Estimation](https://arxiv.org/pdf/1705.00389.pdf)

