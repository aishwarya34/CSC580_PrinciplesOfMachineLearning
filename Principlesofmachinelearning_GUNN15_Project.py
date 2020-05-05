# -*- coding: utf-8 -*-
"""
# CIFAR-10 Image Classification using GUNN-15 (Gradually Updated Neural Networks for Large-Scale Image Recognition)

#### Author: Aishwarya Radhakrishnan
#### Date: March 14, 2020

## Abstract / Introduction

*Convolutional Neural Networks(CNNs) decreases the computation cost for Computer Vision problems than the Deep Neural Networks(DNNs). Increasing the depth leads to increase in parameters of the CNN and hence, increase in computation cost. But as depth plays a vital role in Computer Vision problems, increasing the depth without increasing the computation cost can lead to increased learning. This is achieved by introducing
computation orderings to the channels within convolutional layers.*

*Gradually Updated Neural Networks (GUNN) as opposed to the default Simultaneously Updated Convolutional Network (SUNN / CNN), gradually updates the feature representations against the
traditional convolutional network that computes its output
simultaneously.  This is achieved by updating one channel at a time and using the newly computed parameter of this channel and old parameter of other channels to get another channels in a single convolutional layer. This is repeated for all the channels untill all the old parameters of a single convolutional layer are updated to new values. Thus a single convolutional layer is broken down into multiple gradually updated convolutional layer.*
"""

# Commented out IPython magic to ensure Python compatibility.
import keras
from tensorflow.keras.datasets import cifar10
import numpy as np
import os
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D

from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras import metrics


import tensorflow as tf

tf.keras.backend.clear_session()  # For easy reset of notebook state.

print(tf.__version__)

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

# %matplotlib inline

"""## Loading CIFAR-10 Dataset"""

(X_train_orig, Y_train_orig), (X_test_orig, Y_test_orig) = cifar10.load_data()

X_train_orig = X_train_orig.astype('float32')
X_test_orig = X_test_orig.astype('float32')

# Normalize image vectors
mean = np.mean(X_train_orig,axis=(0,1,2,3))
std = np.std(X_train_orig, axis=(0, 1, 2, 3))
XTRAIN = (X_train_orig-mean)/(std+1e-7)
XTEST = (X_test_orig-mean)/(std+1e-7)

YTRAIN = keras.utils.to_categorical(Y_train_orig, 10)
YTEST = keras.utils.to_categorical(Y_test_orig, 10)

print ("number of training examples = " + str(XTRAIN.shape[0]))
print ("number of test examples = " + str(XTEST.shape[0]))
print ("X_train shape: " + str(XTRAIN.shape))
print ("Y_train shape: " + str(YTRAIN.shape))
print ("X_test shape: " + str(XTEST.shape))
print ("Y_test shape: " + str(YTEST.shape))

X_train_subsetlabel = XTRAIN[np.where((1 == YTRAIN[:,0]) | (YTRAIN[:,1] == 1  ) | (1 ==  YTRAIN[:,2]))]
Y_train_subsetlabel = YTRAIN[np.where((1 == YTRAIN[:,0]) | (YTRAIN[:,1] == 1  ) | (1 ==  YTRAIN[:,2]))]
X_test_subsetlabel = XTEST[np.where((1 == YTEST[:,0]) | (YTEST[:,1] == 1  ) | (1 ==  YTEST[:,2]))]
Y_test_subsetlabel = YTRAIN[np.where((1 == YTEST[:,0]) | (YTEST[:,1] == 1  ) | (1 ==  YTEST[:,2]))]

Y_train_subsetlabel = Y_train_subsetlabel[:, 1:4]
Y_test_subsetlabel = Y_test_subsetlabel[:, 1:4]

print ("number of training examples = " + str(X_train_subsetlabel.shape[0]))
print ("number of test examples = " + str(X_test_subsetlabel.shape[0]))
print ("X_train shape: " + str(X_train_subsetlabel.shape))
print ("Y_train shape: " + str(Y_train_subsetlabel.shape))
print ("X_test shape: " + str(X_test_subsetlabel.shape))
print ("Y_test shape: " + str(Y_test_subsetlabel.shape))

"""<figure>
<center>
<img src='https://drive.google.com/uc?id=1Cn19zTjlJEdYo-EjeGRjPOK9ol_wxfuV' />
<figcaption>CIFAR 10 dataset with 10 classes</figcaption></center>
</figure>

# GUNN Layer implementation (Keras Custom Layer)
"""

def conv_forward(A_shortcut, W1, b1, W2, b2, W3, b3, hparameters):
    """
    Implements the forward propagation for a convolution function
    
    Arguments:
    A_shortcut -- output activations of the previous layer, input to this layer, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    W -- Weights, numpy array of shape (f, f, n_C_prev, n_C)
    b -- Biases, numpy array of shape (1, 1, 1, n_C)
    hparameters -- python dictionary containing "stride" and "pad"
        
    Returns:
    A -- conv output, numpy array of shape (m, n_H, n_W, n_C)
    """
    expand = hparameters["expand"]
    channels = hparameters["channels"]
    depth_batch = channels // expand

    """
    Using conv2d for 1 step of gradual update. 
    Gradually updating by taking `depth_batch` steps of `expand` channels at a time.
    Note: if you dont add b or not use a registered parameter, tensorflow will give error as follows:
    Gradients do not exist for variables ['layer/Variable:0'] when minimizing the loss
    """
    A = tf.identity(A_shortcut)
    for i in range(depth_batch):
        Z = tf.nn.conv2d(A, W1, [1, 1, 1, 1], "VALID") + b1
        A = tf.concat([A[:, :, :, :i*expand ], Z, A[:, :, :, i*expand + expand : ]], 3)
    A = Activation('relu')(A)
  
    for i in range(depth_batch):
        Z = tf.nn.conv2d(A, W2, [1, 1, 1, 1], "SAME") + b2
        A = tf.concat([A[:, :, :, :i*expand ], Z, A[:, :, :, i*expand + expand : ]], 3)
    A = Activation('relu')(A)

    for i in range(channels):
        Z = tf.nn.conv2d(A, W3, [1, 1, 1, 1], "VALID") + b3
        A = tf.concat([A[:, :, :, :i ], Z, A[:, :, :, i + 1 : ]], 3)

    # Add shortcut value to main path. This implements the identity block in Residual Network.
    A = Add()([A , A_shortcut])

    return A



class Gunn2D(layers.Layer):
  """
    Implementation of my own keras layer since the GUNN layer has custom operations having trainable weights.

    __init__(): Input parameters to layer
    build(input_shape) : Weights of Gunn2D layer are defined. Besides trainable weights, you can add non-trainable weights to a layer as well. Such weights are meant not to be taken into account during backpropagation, when you are training the layer.
    call(x) : This is where you write your forward propagation logic. 

    Attributes:
        self.input_channels (int): Input to this layer's number of channels which is not changed by Gunn2D layer.
        self.expansion_rate (int, optional): Update weights of these many channels at once using the whole input.
  """

  def __init__(self, input_channels, expansion_rate=100):
    super(Gunn2D, self).__init__()
    self.input_channels = input_channels
    self.expansion_rate = expansion_rate
    self.hparameters = {"expand": self.expansion_rate, "channels": self.input_channels}

  def build(self, input_shape):
    self.w1 = self.add_weight(shape=(1, 1, self.input_channels, self.expansion_rate), initializer='random_normal', trainable=True)
    self.b1 = self.add_weight(shape=(1, 1, 1, self.expansion_rate), initializer='random_normal', trainable=True)
    self.w2 = self.add_weight(shape=(3, 3, self.input_channels, self.expansion_rate), initializer='random_normal', trainable=True)
    self.b2 = self.add_weight(shape=(1, 1, 1, self.expansion_rate), initializer='random_normal', trainable=True)
    self.w3 = self.add_weight(shape=(1, 1, self.input_channels, 1), initializer='random_normal', trainable=True)
    self.b3 = self.add_weight(shape=(1, 1, 1, 1), initializer='random_normal', trainable=True)

  def call(self, inputs):
    output = conv_forward(inputs, self.w1, self.b1, self.w2, self.b2, self.w3, self.b3, self.hparameters)
    return output

"""## Building GUNN-15 Model in Keras for 10 classes of CIFAR-10 dataset"""

def GunnModel(input_shape):
    """
    Implementation of the GUNN-15 Model.

    GunnModel implements image classification model for CIFAR-10 dataset.

    Gunn2D is used to replace convolutional layers. Residual Networks principles are also used in Gunn2D layer.
    Conv2D is used for Convolutional Neural Networks - Forward and Backward pass
    BatchNormalization and Activation are used.
    
    Arguments:
    input_shape -- shape of the images of the dataset

    Returns:
    model -- a Model() instance in Keras
    """
    
    X_input = Input(input_shape)
    X = Conv2D(64, (3, 3), strides = (1, 1), padding='same', name = 'z1')(X_input) # 32x32x3 -> 32x32x64   ; padding = 1
    X = BatchNormalization(axis = 3 , name = 'bn1')(X)
    X = Activation('relu')(X)
    convlayer = Conv2D(240, (1, 1), strides = (1, 1), padding='valid', name = 'z2')
    X = convlayer(X) # 32x32x64 -> 32x32x240
    layer = BatchNormalization(axis = 3 , name = 'bn2')
    X = layer(X)
    X = Activation('relu')(X)
    X = Gunn2D(240, 20)(X) # custom Keras layer class
    X = Conv2D(300, (1, 1), strides = (1, 1), padding='valid', name = 'z3')(X)
    X = BatchNormalization(axis = 3 , name = 'bn3')(X)
    X = Activation('relu')(X)
    X = AveragePooling2D((2, 2), name = 'avg_pool1')(X)
    X = Gunn2D(300, 20)(X)
    X = Conv2D(360, (1, 1), strides = (1, 1), padding='valid', name = 'z4')(X)
    X = BatchNormalization(axis = 3 , name = 'bn4')(X)
    X = Activation('relu')(X)
    X = AveragePooling2D((2, 2), name = 'avg_pool2')(X)
    X = Gunn2D(360, 20)(X)
    X = Conv2D(360, (1, 1), strides = (1, 1), padding='valid', name = 'z5')(X)
    X = BatchNormalization(axis = 3 , name = 'bn5')(X)
    X = Activation('relu')(X)
    X = AveragePooling2D((8, 8), name = 'avg_pool3')(X)
    X = Flatten()(X)
    X = Dense(360, activation='softmax', name = 'fc1')(X)
    X = Dense(360, activation='softmax', name = 'fc2')(X)
    X = Dense(3, activation='softmax', name = 'fc3')(X)

    model = Model(inputs = X_input, outputs = X, name = 'GUNN-15-Model')
    return model

"""## Fit model on CIFAR-10 dataset"""

X_train = X_train_subsetlabel[:5000]
Y_train = Y_train_subsetlabel[:5000]
X_test = X_test_subsetlabel[:100]
Y_test = Y_test_subsetlabel[:100]

gunnModel = GunnModel(X_train.shape[1:])
gunnModel.compile(optimizer = "adam", loss='categorical_crossentropy', metrics=[metrics.categorical_accuracy])

gunnModel.fit(x = X_train , y = Y_train, epochs = 100, steps_per_epoch = (X_train.shape[0]//50))
preds = gunnModel.evaluate(x=X_test, y=Y_test)
print()
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))

gunnModel.summary()

tf.keras.utils.plot_model(gunnModel, 'gunnModel_model.png', show_shapes=True)

"""# Write output to file"""

f = open("results.txt", "a")
f.write("\nLoss = " + str(preds[0]) +"\n")
f.write("Test Accuracy = " + str(preds[1]))
f.close()
