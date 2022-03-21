# CS598-Deep-Learning-MPs
CS 598 Deep Learning, UIUC Machine Problems 

**Problem 1: Implementation of fully connected neural network from scratch using numpy**

Implement and train a neural network from scratch in Python for the MNIST dataset (no PyTorch). The neural network should be trained on the Training Set using stochastic gradient descent. Target accuracy on the test set: 97-98%

**Problem 2: Implementation of Convolution Neural Network from scratch using numpy**

Implement and train a convolution neural network from scratch in Python for the MNIST dataset (no PyTorch). The convolution network should have a single hidden layer with multiple channels. Target accuracy on the test set: 94% 

**Problem 3: Implementation of Deep CNN for CIFAR 10**

Train a deep convolution network on a GPU with PyTorch for the CIFAR10 dataset. The convolution network should use dropout, trained with RMSprop or ADAM, and data augmentation. Also, compare dropout test accuracy using the heuristic prediction rule and Monte Carlo simulation.

**Problem 4: Implementation of Deep Resnet on CIFAR 100**

Implement a deep residual neural network for CIFAR100 using your own Resnet and pretrained Resnet implemented in Pytorch. Target test set accuracy: 60% for own Resnet implementation and 70% for pretrained Resnet implementation.

**Problem 5: Image Similarity with Deep Ranking on Tiny Imagenet**

The goal of this project is to introduce you to the computer vision task of image similarity. The task of image similarity is retrieve a set of n images closest to the query image. One application of this task could involve visual search engine where we provide a query image and
want to find an image closest that image in the database.

**Problem 6: Implementation of GAN with Wasserstien Gradient Penalty (Wasserstien GAN) and Auxilliary Classifier (ACGAN) on CIFAR 10**

The objective is to train a discriminator and generator pair to generate artificial data  similar to the 32 x 32 dimensional coloured image from the corresponding one of the 10 mutually exclusive classes namely fairplane, automobile, bird, cat, deer, dog, frog, horse, ship, truckg in CIFAR10 data set. The CIFAR10 data set has 50000 coloured training images and 10000 images in the test set of dimensions 3 x 32 x 32.

**Problem 7: Sentiment Analysis for IMDB Movie Reviews**

This assignment will work with the Large Movie Review Dataset and provide an understanding of how Deep Learning is used within the field of Natural Language Processing (NLP). We will train models to detect the sentiment of written text. More specifically, we will try to feed a movie review into a model and have it predict whether it is a positive or negative movie review.

The assignment starts off with how to preprocess the dataset into a more appropriate format followed by three main parts. Each part will involve training different types of models and will detail what you need to do as well as what needs to be turned in. The assignment is meant to provide you with a working knowledge of how to use neural networks with sequential data.
* Part one deals with training basic **Bag of Words** models.
* Part two will start incoporating **temporal information by using LSTM** layers.
* Part three will show how to **train a language model** and how doing this as a first step can sometimes improve results for other tasks

**Final Project: Show and Tell Neural Captioning**
Code available on https://github.com/kaurrachneet6/ShowAndTell-neural-captioning
