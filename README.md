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

The Keras custom layer needs forward propagation definition only.  Since our custom layer has trainable weights, we need to use stateful custom operations using custom layer class definition. 

Note: If you want to build a keras model with a custom layer that performs a custom operation and has a custom gradient, you should use @tf.custom_gradient.


## Building GUNN-15 Model in Keras for 10 classes of CIFAR-10 dataset

Using Keras Convolutional Neural Networks building blocks and custom implemented Gunn2D layer created GUNN-15 Model.

<figure>
<div align="center">
<img src='https://github.com/aishwarya34/CSC580_PrinciplesOfMachineLearning/blob/master/img/GunnModel.png' /><br>
<figcaption>Figure 4: Gunn Model. The Gunn Model has Gunn2D layer implemented at the layers marked as * i.e. Conv2*, Conv3* and Conv4* layers. </figcaption></div>
</figure>




Built Gradually updated Convolutional Neural Net following GUNN-15 model having 15 layers that classifies CIFAR-10's 3 classes in Colab with the following architecture:

- Training examples : 5000  
- Testing examples : 100  
- Batch Size : 50 
- epochs : 50
- Adam optimizer : for adaptive estimates of lower-order moments
- loss : Categorical crossentropy for multiple class estimation
- activation layer : RELU
- Batch Normalization : for the network to use higher learning rate without vanishing or exploding gradients. 

In addition, all the layers parameters are defined in accordance with the model in figure 4.



## Training

<figure>
<div align="center">
<img src='https://github.com/aishwarya34/CSC580_PrinciplesOfMachineLearning/blob/master/img/GUNN-15_model.png' /><br>
<figcaption>Figure 5: GUNN-15 model having custon Gunn2D layer. </figcaption></div>
</figure>


The above figure shows the High-Level Keras Automatic Differentiation graph having different layers like Conv2d, Activation etc that are Keras APIs as well as the **Gunn2D layer** which is a custom layer.



Trained the Convolutional Neural Net to classify CIFAR-10's 3 classes in Colab using 5000 training examples and 100 testing examples with batch size of 50 and for 50 epochs, I get the following  output.


Epoch 1/50
Resnet : (20, 32, 32, 240)
Resnet : (20, 16, 16, 300)
Resnet : (20, 8, 8, 360)
Resnet : (20, 32, 32, 240)
Resnet : (20, 16, 16, 300)
Resnet : (20, 8, 8, 360)
50/50 [==============================] - 1432s 29s/step - loss: 2.2360 - categorical_accuracy: 0.3160
Epoch 2/50
50/50 [==============================] - 1398s 28s/step - loss: 2.1116 - categorical_accuracy: 0.3550
Epoch 3/50
50/50 [==============================] - 1388s 28s/step - loss: 1.9972 - categorical_accuracy: 0.3550
Epoch 4/50
50/50 [==============================] - 1398s 28s/step - loss: 1.8943 - categorical_accuracy: 0.3550
Epoch 5/50
50/50 [==============================] - 1393s 28s/step - loss: 1.8017 - categorical_accuracy: 0.3550
Epoch 6/50
50/50 [==============================] - 1384s 28s/step - loss: 1.7186 - categorical_accuracy: 0.3550
Epoch 7/50
50/50 [==============================] - 1377s 28s/step - loss: 1.6449 - categorical_accuracy: 0.3550
Epoch 8/50
50/50 [==============================] - 1373s 27s/step - loss: 1.5792 - categorical_accuracy: 0.3550
Epoch 9/50
50/50 [==============================] - 1385s 28s/step - loss: 1.5204 - categorical_accuracy: 0.3550
Epoch 10/50
50/50 [==============================] - 1388s 28s/step - loss: 1.4700 - categorical_accuracy: 0.3550
Epoch 11/50
50/50 [==============================] - 1381s 28s/step - loss: 1.4231 - categorical_accuracy: 0.3550
Epoch 12/50
50/50 [==============================] - 1384s 28s/step - loss: 1.3849 - categorical_accuracy: 0.3550
Epoch 13/50
50/50 [==============================] - 1382s 28s/step - loss: 1.3457 - categorical_accuracy: 0.3550
Epoch 14/50
50/50 [==============================] - 1403s 28s/step - loss: 1.3118 - categorical_accuracy: 0.4270
Epoch 15/50
50/50 [==============================] - 1398s 28s/step - loss: 1.2808 - categorical_accuracy: 0.5560
Epoch 16/50
50/50 [==============================] - 1384s 28s/step - loss: 1.2505 - categorical_accuracy: 0.5540
Epoch 17/50
50/50 [==============================] - 1390s 28s/step - loss: 1.2204 - categorical_accuracy: 0.5610
Epoch 18/50
50/50 [==============================] - 1376s 28s/step - loss: 1.1898 - categorical_accuracy: 0.5550
Epoch 19/50
50/50 [==============================] - 1385s 28s/step - loss: 1.1730 - categorical_accuracy: 0.5250
Epoch 20/50
50/50 [==============================] - 1389s 28s/step - loss: 1.1443 - categorical_accuracy: 0.5350
Epoch 21/50
50/50 [==============================] - 1380s 28s/step - loss: 1.0989 - categorical_accuracy: 0.5760
Epoch 22/50
50/50 [==============================] - 1390s 28s/step - loss: 1.0757 - categorical_accuracy: 0.5610
Epoch 23/50
50/50 [==============================] - 1383s 28s/step - loss: 1.0449 - categorical_accuracy: 0.5810
Epoch 24/50
50/50 [==============================] - 1393s 28s/step - loss: 1.0258 - categorical_accuracy: 0.5790
Epoch 25/50
50/50 [==============================] - 1386s 28s/step - loss: 1.0040 - categorical_accuracy: 0.5730
Epoch 26/50
50/50 [==============================] - 1378s 28s/step - loss: 0.9987 - categorical_accuracy: 0.5620
Epoch 27/50
50/50 [==============================] - 1389s 28s/step - loss: 0.9984 - categorical_accuracy: 0.5480
Epoch 28/50
50/50 [==============================] - 1388s 28s/step - loss: 0.9607 - categorical_accuracy: 0.5640
Epoch 29/50
50/50 [==============================] - 1379s 28s/step - loss: 0.9398 - categorical_accuracy: 0.5770
Epoch 30/50
50/50 [==============================] - 1368s 27s/step - loss: 0.9290 - categorical_accuracy: 0.5790
Epoch 31/50
30/50 [=================>............] - ETA: 8:56 - loss: 0.9170 - categorical_accuracy: 0.5783


Epoch 1/10
Resnet : (10, 32, 32, 240)
Resnet : (10, 16, 16, 300)
Resnet : (10, 8, 8, 360)
10/10 [==============================] - 140s 14s/step - loss: 1.4316 - categorical_accuracy: 0.4400
Epoch 2/10
10/10 [==============================] - 138s 14s/step - loss: 1.4236 - categorical_accuracy: 0.4700
Epoch 3/10
10/10 [==============================] - 135s 14s/step - loss: 1.4094 - categorical_accuracy: 0.4500
Epoch 4/10
10/10 [==============================] - 137s 14s/step - loss: 1.3995 - categorical_accuracy: 0.5700
Epoch 5/10
10/10 [==============================] - 134s 13s/step - loss: 1.3906 - categorical_accuracy: 0.5400
Epoch 6/10
10/10 [==============================] - 134s 13s/step - loss: 1.3823 - categorical_accuracy: 0.5000
Epoch 7/10
10/10 [==============================] - 135s 13s/step - loss: 1.3742 - categorical_accuracy: 0.4900
Epoch 8/10
10/10 [==============================] - 133s 13s/step - loss: 1.3639 - categorical_accuracy: 0.5000
Epoch 9/10
10/10 [==============================] - 134s 13s/step - loss: 1.3535 - categorical_accuracy: 0.5600
Epoch 10/10
10/10 [==============================] - 135s 13s/step - loss: 1.3436 - categorical_accuracy: 0.5800
Resnet : (None, 32, 32, 240)
Resnet : (None, 16, 16, 300)
Resnet : (None, 8, 8, 360)
1/1 [==============================] - 0s 1ms/step - loss: 2.7686 - categorical_accuracy: 0.0000e+00

Loss = 2.7685534954071045
Test Accuracy = 0.0


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




