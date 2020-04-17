<div align="center">
    <h1 style="font-size:300%;">Gradually Updated Neural Networks for Large-Scale Image Recognition</h1>
    <h3 style="font-size:100%;">Aishwarya Radhakrishnan <br>
    March 14, 2020</h3>
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
<figcaption>Figure 1: CIFAR 10 dataset with 10 classes that are randomly chosen</figcaption></div>
</figure>



## GUNN Layer implementation (Keras Custom Layer)

Keras Custom Layer needs to be used because GUNN layer consist of convolutional network which decomposes
the original computation into multiple sequential channel-wise convolutional operations as opposed to single convolutional operations in Conv2D.


<figure>
<div align="center">
<img src='https://github.com/aishwarya34/CSC580_PrinciplesOfMachineLearning/blob/master/img/SUNNvsGUNN.png' /><br>
<figcaption>Figure 2: SUNN vs GUNN. SUNN architecture is our usually used Convolution Layer while GUNN is the channel-wise decomposed architecture.</figcaption></div>
</figure>


The expansion rate is used to know how many channels are simultaneously updated when decomposing the channels. The kernel channel is the same as the expansion rate. The kernel size is given by fxf.

<figure>
<div align="center">
<img src='https://github.com/aishwarya34/CSC580_PrinciplesOfMachineLearning/blob/master/img/GunnLayer.png' /><br>
<figcaption>Figure 3: Gunn2D Keras custom layer. The Gunn2D layer expands channel at K rate and also uses Residual Network identity block implementation. </figcaption></div>
</figure>

<br>

The Keras custom layer needs forward propagation definition only. 

Note: If you want to build a keras model with a custom layer that performs a custom operation and has a custom gradient, you should use @tf.custom_gradient.


## Building GUNN-15 Model in Keras for 10 classes of CIFAR-10 dataset

Using Keras Convolutional Neural Networks building blocks and custom implemented Gunn2D layer created GUNN-15 Model.

<figure>
<div align="center">
<img src='https://github.com/aishwarya34/CSC580_PrinciplesOfMachineLearning/blob/master/img/GunnModel.png' /><br>
<figcaption>Figure 4: Gunn Model. The Gunn Model has Gunn2D layer implemented at the layers marked as * i.e. Conv2*, Conv3* and Conv4* layers. </figcaption></div>
</figure>



## Training


<figure>
<div align="center">
<img src='https://github.com/aishwarya34/CSC580_PrinciplesOfMachineLearning/blob/master/img/GUNN-15_model.png' /><br>
<figcaption>Figure 5: GUNN-15 model having custon Gunn2D layer. </figcaption></div>
</figure>


## Evaluation

## Results / Conclusion

## Reference

[1] GUNN is used to replace convolutional layers which is stated in paper:  Qiao, Siyuan et al. “Gradually Updated Neural Networks for Large-Scale Image Recognition.” ICML (2017).

[2]  The principle of Residual Networks also applies. He, K., Zhang, X., Ren, S., and Sun, J. Deep residual
learning for image recognition. IEEE Conference on
Computer Vision and Pattern Recognition, 2016a.

[3]  The principle of Residual Networks also applies. He, K., Zhang, X., Ren, S., and Sun, J. Identity mappings
in deep residual networks. ECCV, 2016b.

[4]  Batch Normalization will be implemented. Ioffe, S. and Szegedy, C. Batch normalization: Accelerating
deep network training by reducing internal covariate shift.
In International Conference on Machine Learning, 2015.




