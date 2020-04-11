# CSC 580 Principles Of Machine Learning : Computer Vision Project


## Gradually Updated Neural Networks for Large-Scale Image Recognition
#### Author: Aishwarya Radhakrishnan
#### Date: March 14, 2020


<div align="center">
    <h1 style="font-size:300%;">Gradually Updated Neural Networks for Large-Scale Image Recognition</h1>
</div>
  
<hr>



## Abstract / Introduction

*Convolutional Neural Networks(CNNs) decreases the computation cost for Computer Vision problems than the Deep Neural Networks(DNNs). Increasing the depth leads to increase in parameters of the CNN and hence, increase in computation cost. But as depth plays a vital role in Computer Vision problems, increasing the depth without increasing the computation cost can lead to increased learning. This is achieved by introducing
computation orderings to the channels within convolutional layers.*

*Gradually Updated Neural Networks (GUNN) as opposed to the default Simultaneously Updated Convolutional Network (SUNN / CNN), gradually updates the feature representations against the
traditional convolutional network that computes its output
simultaneously.  This is achieved by updating one channel at a time and using the newly computed parameter of this channel and old parameter of other channels to get another channels in a single convolutional layer. This is repeated for all the channels untill all the old parameters of a single convolutional layer are updated to new values. Thus a single convolutional layer is broken down into multiple gradually updated convolutional layer.*



## Prerequisites

Installing pip, tensorflow

## CIFAR-10 Dataset

Number of Training examples = 50000 <br>
Number of Test examples = 10000 <br>
X_train shape: (50000, 32, 32, 3) <br>
Y_train shape: (50000, 10) <br>
X_test shape: (10000, 32, 32, 3) <br>
Y_test shape: (10000, 10)


<figure>
<div align="center">
<img src='https://github.com/aishwarya34/CSC580_PrinciplesOfMachineLearning/blob/master/img/CIFAR10.png' /><br>
<figcaption>CIFAR 10 dataset with 10 classes</figcaption></div>
</figure>

